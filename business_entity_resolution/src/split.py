"""Freeze a country- and match-count-stratified Source 1 holdout.

The holdout is 10% of training Source 1 entities. Sampling is deterministic
given ``config.RANDOM_SEED``. A 50,000-entity dev sample is drawn from the
holdout with the same stratification, for the first blocking measurement.
"""

from collections import defaultdict
import csv
import random

from . import config

HOLDOUT_FRACTION = 0.10
DEV_SAMPLE_SIZE = 50_000
SPLIT_PATH = config.REPORTS_DIR / "validation_split.tsv"
SUMMARY_PATH = config.REPORTS_DIR / "validation_split_summary.tsv"


def match_bucket(count: int) -> str:
    return "8+" if count >= 8 else str(count)


def _read_countries() -> dict[str, str]:
    countries: dict[str, str] = {}
    with config.TRAIN_SOURCE1_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        id_idx = header.index("entity_id")
        country_idx = header.index("country")
        for row in reader:
            if len(row) != len(header):
                continue
            countries[row[id_idx]] = row[country_idx].strip() or "<EMPTY>"
    return countries


def _read_match_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with config.TRAIN_GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        id_idx = header.index("source1_entity_id")
        match_idx = header.index("matched_entity_ids")
        for row in reader:
            if len(row) != len(header):
                continue
            raw = row[match_idx].strip()
            counts[row[id_idx]] = 0 if not raw else len(set(raw.split(",")))
    return counts


def build_split() -> str:
    countries = _read_countries()
    counts = _read_match_counts()
    missing_labels = [entity_id for entity_id in countries if entity_id not in counts]
    if missing_labels:
        raise RuntimeError(f"{len(missing_labels)} Source 1 entities have no ground-truth row")

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entity_id, country in countries.items():
        groups[(country, match_bucket(counts[entity_id]))].append(entity_id)

    rng = random.Random(config.RANDOM_SEED)
    assignments: dict[str, str] = {}
    summary_rows: list[tuple[str, str, int, int, int]] = []
    for (country, bucket), entity_ids in sorted(groups.items()):
        ordered = sorted(entity_ids)
        rng.shuffle(ordered)
        holdout_n = int(round(len(ordered) * HOLDOUT_FRACTION))
        if len(ordered) >= 10 and holdout_n == 0:
            holdout_n = 1
        holdout_n = min(holdout_n, len(ordered))
        for entity_id in ordered[:holdout_n]:
            assignments[entity_id] = "valid"
        for entity_id in ordered[holdout_n:]:
            assignments[entity_id] = "train"
        summary_rows.append((country, bucket, len(ordered), holdout_n, len(ordered) - holdout_n))

    dev_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entity_id, split_name in assignments.items():
        if split_name == "valid":
            dev_groups[(countries[entity_id], match_bucket(counts[entity_id]))].append(entity_id)
    valid_total = sum(len(ids) for ids in dev_groups.values())
    dev_ids: set[str] = set()
    dev_rng = random.Random(config.RANDOM_SEED)
    for key, entity_ids in sorted(dev_groups.items()):
        ordered = sorted(entity_ids)
        dev_rng.shuffle(ordered)
        take = int(round(len(ordered) * DEV_SAMPLE_SIZE / valid_total))
        if ordered and take == 0 and len(dev_ids) < DEV_SAMPLE_SIZE:
            take = 1
        dev_ids.update(ordered[:take])
    if len(dev_ids) > DEV_SAMPLE_SIZE:
        dev_ids = set(sorted(dev_ids)[:DEV_SAMPLE_SIZE])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with SPLIT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "country", "match_count", "bucket", "split", "dev_sample"])
        for entity_id in sorted(countries):
            writer.writerow([
                entity_id,
                countries[entity_id],
                counts[entity_id],
                match_bucket(counts[entity_id]),
                assignments[entity_id],
                "1" if entity_id in dev_ids else "0",
            ])
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["country", "bucket", "entities", "valid", "train"])
        for row in summary_rows:
            writer.writerow(row)

    valid_n = sum(row[3] for row in summary_rows)
    lines = [
        f"Seed: {config.RANDOM_SEED}",
        f"Holdout fraction: {HOLDOUT_FRACTION:.0%}",
        f"Source 1 entities: {len(countries):,}",
        f"Validation entities: {valid_n:,}",
        f"Train entities: {len(countries) - valid_n:,}",
        f"Dev sample (subset of validation): {len(dev_ids):,}",
        f"Strata: {len(summary_rows)}",
        f"Split file: {SPLIT_PATH}",
        f"Summary file: {SUMMARY_PATH}",
    ]
    return "\n".join(lines)


def main() -> int:
    report = build_split()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
