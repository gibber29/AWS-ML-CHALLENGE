"""Exact, streaming audit of the seven challenge TSV files."""

from collections import Counter
from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Iterable

from . import config
from .data_loader import ENTITY_COLUMNS, GROUND_TRUTH_COLUMNS, DataValidationError
from .normalize import normalize_address, normalize_name


@dataclass
class FileAudit:
    key: str
    path: Path
    columns: list[str]
    rows: int = 0
    malformed_rows: int = 0
    duplicate_ids: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    malformed_ids: int = 0
    missing_names: int = 0
    missing_addresses: int = 0
    countries: Counter[str] = field(default_factory=Counter)
    id_numbers: set[int] | None = None


def _id_number(entity_id: str, expected_prefix: str) -> int | None:
    prefix = expected_prefix + "-"
    suffix = entity_id[len(prefix):] if entity_id.startswith(prefix) else ""
    return int(suffix) if suffix.isdigit() else None


def _reader(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    handle = path.open("r", encoding="utf-8", newline="")
    return handle, csv.reader(handle, delimiter="\t")


def _scan_entity_file(key: str, path: Path, prefix: str, retain_ids: bool) -> FileAudit:
    handle, reader = _reader(path)
    with handle:
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise DataValidationError(f"Empty TSV file: {path}") from exc
        missing = [column for column in ENTITY_COLUMNS if column not in columns]
        if missing:
            raise DataValidationError(f"{path} missing {missing}; actual columns: {columns}")
        indexes = {column: columns.index(column) for column in ENTITY_COLUMNS}
        result = FileAudit(key, path, columns, id_numbers=set() if retain_ids else None)
        seen: set[int] = set()
        malformed_seen: set[str] = set()
        for row in reader:
            result.rows += 1
            if len(row) != len(columns):
                result.malformed_rows += 1
                continue
            entity_id = row[indexes["entity_id"]]
            number = _id_number(entity_id, prefix)
            duplicate = False
            if number is None:
                result.malformed_ids += 1
                duplicate = entity_id in malformed_seen
                malformed_seen.add(entity_id)
            else:
                duplicate = number in seen
                seen.add(number)
                if retain_ids:
                    result.id_numbers.add(number)  # type: ignore[union-attr]
            if duplicate:
                result.duplicate_ids += 1
                if len(result.duplicate_examples) < 5:
                    result.duplicate_examples.append(entity_id)
            if not row[indexes["business_name"]].strip():
                result.missing_names += 1
            if not row[indexes["business_address"]].strip():
                result.missing_addresses += 1
            result.countries[row[indexes["country"]].strip() or "<EMPTY>"] += 1
        return result


@dataclass
class GroundTruthAudit:
    columns: list[str]
    rows: int = 0
    duplicate_source1_rows: int = 0
    duplicate_source1_examples: list[str] = field(default_factory=list)
    distribution: Counter[str] = field(default_factory=Counter)
    s2_matches: int = 0
    s3_matches: int = 0
    duplicate_ids_in_lists: int = 0
    duplicate_list_rows: int = 0
    invalid_source1_references: int = 0
    missing_target_references: int = 0
    source1_ids_as_matches: int = 0
    malformed_match_ids: int = 0
    missing_ground_truth_rows: int = 0
    anomaly_examples: list[str] = field(default_factory=list)
    sample_pairs: list[tuple[str, str]] = field(default_factory=list)


def _scan_ground_truth(
    path: Path, source1: set[int], source2: set[int], source3: set[int]
) -> GroundTruthAudit:
    handle, reader = _reader(path)
    with handle:
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise DataValidationError(f"Empty TSV file: {path}") from exc
        missing = [column for column in GROUND_TRUTH_COLUMNS if column not in columns]
        if missing:
            raise DataValidationError(f"{path} missing {missing}; actual columns: {columns}")
        s1_col = columns.index("source1_entity_id")
        matches_col = columns.index("matched_entity_ids")
        result = GroundTruthAudit(columns)
        seen_s1: set[int] = set()
        malformed_s1_seen: set[str] = set()
        for row in reader:
            result.rows += 1
            if len(row) != len(columns):
                result.invalid_source1_references += 1
                if len(result.anomaly_examples) < 10:
                    result.anomaly_examples.append(f"malformed ground-truth row {result.rows + 1}")
                continue
            s1_id = row[s1_col]
            s1_number = _id_number(s1_id, "S1")
            if s1_number is None:
                duplicate_s1 = s1_id in malformed_s1_seen
                malformed_s1_seen.add(s1_id)
                result.invalid_source1_references += 1
            else:
                duplicate_s1 = s1_number in seen_s1
                seen_s1.add(s1_number)
                if s1_number not in source1:
                    result.invalid_source1_references += 1
                    if len(result.anomaly_examples) < 10:
                        result.anomaly_examples.append(f"unknown Source 1 reference: {s1_id}")
            if duplicate_s1:
                result.duplicate_source1_rows += 1
                if len(result.duplicate_source1_examples) < 5:
                    result.duplicate_source1_examples.append(s1_id)

            raw_matches = row[matches_col]
            items = raw_matches.split(",") if raw_matches else []
            duplicate_count = len(items) - len(set(items))
            if duplicate_count:
                result.duplicate_list_rows += 1
                result.duplicate_ids_in_lists += duplicate_count
            unique_items = set(items)
            count_valid_targets = 0
            sample_candidate: tuple[str, str] | None = None
            for match_id in sorted(unique_items):
                if match_id.startswith("S1-"):
                    result.source1_ids_as_matches += 1
                    continue
                if match_id.startswith("S2-"):
                    match_number = _id_number(match_id, "S2")
                    if match_number is None:
                        result.malformed_match_ids += 1
                    else:
                        result.s2_matches += 1
                        count_valid_targets += 1
                        if match_number not in source2:
                            result.missing_target_references += 1
                        elif sample_candidate is None:
                            sample_candidate = (s1_id, match_id)
                    continue
                if match_id.startswith("S3-"):
                    match_number = _id_number(match_id, "S3")
                    if match_number is None:
                        result.malformed_match_ids += 1
                    else:
                        result.s3_matches += 1
                        count_valid_targets += 1
                        if match_number not in source3:
                            result.missing_target_references += 1
                        elif sample_candidate is None:
                            sample_candidate = (s1_id, match_id)
                    continue
                result.malformed_match_ids += 1
            if sample_candidate is not None and len(result.sample_pairs) < 8:
                result.sample_pairs.append(sample_candidate)
            bucket = "zero" if count_valid_targets == 0 else "one" if count_valid_targets == 1 else "multiple"
            result.distribution[bucket] += 1
        result.missing_ground_truth_rows = len(source1 - seen_s1)
        return result


def _collect_sample_records(paths: Iterable[Path], wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    found: dict[str, tuple[str, str, str]] = {}
    for path in paths:
        handle, reader = _reader(path)
        with handle:
            columns = next(reader)
            indexes = {name: columns.index(name) for name in ENTITY_COLUMNS}
            for row in reader:
                if len(row) != len(columns):
                    continue
                entity_id = row[indexes["entity_id"]]
                if entity_id in wanted:
                    found[entity_id] = (
                        row[indexes["business_name"]], row[indexes["business_address"]],
                        row[indexes["country"]],
                    )
                    if len(found) == len(wanted):
                        return found
    return found


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def generate_audit() -> str:
    specs = [
        ("train_source1", config.TRAIN_SOURCE1_PATH, "S1", True),
        ("train_source2", config.TRAIN_SOURCE2_PATH, "S2", True),
        ("train_source3", config.TRAIN_SOURCE3_PATH, "S3", True),
        ("test_source1", config.TEST_SOURCE1_PATH, "S1", False),
        ("test_source2", config.TEST_SOURCE2_PATH, "S2", False),
        ("test_source3", config.TEST_SOURCE3_PATH, "S3", False),
    ]
    audits = {key: _scan_entity_file(key, path, prefix, retain) for key, path, prefix, retain in specs}
    gt = _scan_ground_truth(
        config.TRAIN_GROUND_TRUTH_PATH,
        audits["train_source1"].id_numbers or set(),
        audits["train_source2"].id_numbers or set(),
        audits["train_source3"].id_numbers or set(),
    )

    sample_ids = {item for pair in gt.sample_pairs for item in pair}
    sample_records = _collect_sample_records(
        (config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH),
        sample_ids,
    )
    lines = ["BUSINESS ENTITY RESOLUTION DATA AUDIT", "=" * 43, ""]
    lines.append("FILES AND SCHEMAS")
    for key in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
        item = audits[key]
        lines.extend([
            f"- {key}: {_fmt_int(item.rows)} rows x {len(item.columns)} columns",
            f"  columns: {item.columns}",
            f"  duplicate entity IDs: {_fmt_int(item.duplicate_ids)}",
            f"  malformed rows / IDs: {_fmt_int(item.malformed_rows)} / {_fmt_int(item.malformed_ids)}",
            f"  empty business names / addresses: {_fmt_int(item.missing_names)} / {_fmt_int(item.missing_addresses)}",
            "  records by country: " + ", ".join(f"{k}={_fmt_int(v)}" for k, v in sorted(item.countries.items())),
        ])
        if item.duplicate_examples:
            lines.append(f"  duplicate examples: {item.duplicate_examples}")
    lines.extend([
        f"- train_ground_truth: {_fmt_int(gt.rows)} rows x {len(gt.columns)} columns",
        f"  columns: {gt.columns}", "",
    ])

    train_countries = set().union(*(audits[f"train_source{i}"].countries for i in (1, 2, 3)))
    test_countries = set().union(*(audits[f"test_source{i}"].countries for i in (1, 2, 3)))
    lines.extend([
        "COUNTRY COMPARISON",
        f"- Training countries observed: {sorted(train_countries)}",
        f"- Test countries observed: {sorted(test_countries)}",
        f"- Test-only countries: {sorted(test_countries - train_countries)}",
    ])
    train_france = sum(audits[f"train_source{i}"].countries.get("France", 0) for i in (1, 2, 3))
    test_france = sum(audits[f"test_source{i}"].countries.get("France", 0) for i in (1, 2, 3))
    lines.append(
        f"- France observation: train={_fmt_int(train_france)}, test={_fmt_int(test_france)}. "
        + ("France is present only in test." if test_france and not train_france else "France is not exclusively test-only in the observed data.")
    )

    zero = gt.distribution["zero"]
    singleton_pct = (100 * zero / gt.rows) if gt.rows else 0.0
    lines.extend([
        "", "GROUND TRUTH",
        f"- Source 1 singleton rows (zero matches): {_fmt_int(zero)} ({singleton_pct:.2f}%)",
        f"- Match-count distribution: zero={_fmt_int(zero)}, one={_fmt_int(gt.distribution['one'])}, multiple={_fmt_int(gt.distribution['multiple'])}",
        f"- Source 2 / Source 3 match references: {_fmt_int(gt.s2_matches)} / {_fmt_int(gt.s3_matches)}",
        f"- Duplicate Source 1 rows: {_fmt_int(gt.duplicate_source1_rows)}",
        f"- Duplicate IDs inside match lists: {_fmt_int(gt.duplicate_ids_in_lists)} across {_fmt_int(gt.duplicate_list_rows)} rows",
        f"- Invalid Source 1 references: {_fmt_int(gt.invalid_source1_references)}",
        f"- Source 1 entities missing a ground-truth row: {_fmt_int(gt.missing_ground_truth_rows)}",
        f"- Matched IDs absent from Source 2/3: {_fmt_int(gt.missing_target_references)}",
        f"- Source 1 IDs incorrectly included as matches: {_fmt_int(gt.source1_ids_as_matches)}",
        f"- Malformed match IDs: {_fmt_int(gt.malformed_match_ids)}",
    ])
    if gt.duplicate_source1_examples:
        lines.append(f"- Duplicate Source 1 examples: {gt.duplicate_source1_examples}")
    if gt.anomaly_examples:
        lines.append(f"- Other anomaly examples: {gt.anomaly_examples}")

    lines.extend(["", "ANONYMIZATION-SAFE VARIATION EXAMPLES"])
    shown = 0
    for s1_id, match_id in gt.sample_pairs:
        if s1_id not in sample_records or match_id not in sample_records:
            continue
        left_name, left_address, left_country = sample_records[s1_id]
        right_name, right_address, right_country = sample_records[match_id]
        ln, rn = normalize_name(left_name), normalize_name(right_name)
        la, ra = normalize_address(left_address), normalize_address(right_address)
        shown += 1
        lines.append(
            f"- Example {shown} ({left_country}->{right_country}, target={match_id[:2]}): "
            f"name chars {len(left_name)}->{len(right_name)}, core-token Jaccard "
            f"{_jaccard(ln.core.split(), rn.core.split()):.2f}, accent-fold exact={ln.core_accent_folded == rn.core_accent_folded}; "
            f"address chars {len(left_address)}->{len(right_address)}, token Jaccard "
            f"{_jaccard(la.tokens, ra.tokens):.2f}, numeric overlap={sorted(set(la.numeric_tokens) & set(ra.numeric_tokens))}"
        )
        if shown == 5:
            break
    if not shown:
        lines.append("- No valid matched pairs were available for examples.")
    lines.append("- Raw business names and addresses are intentionally omitted from this report.")

    train_cartesian = audits["train_source1"].rows * (audits["train_source2"].rows + audits["train_source3"].rows)
    test_cartesian = audits["test_source1"].rows * (audits["test_source2"].rows + audits["test_source3"].rows)
    lines.extend([
        "", "RAW CARTESIAN COMPARISONS",
        f"- Train: {_fmt_int(train_cartesian)}",
        f"- Test: {_fmt_int(test_cartesian)}",
        f"- Combined: {_fmt_int(train_cartesian + test_cartesian)}",
        "", "Audit completed using only the provided local TSV files.",
    ])
    return "\n".join(lines)


def main() -> int:
    report = generate_audit()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.REPORTS_DIR / "data_audit.txt"
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
