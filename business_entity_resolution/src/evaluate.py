"""Official entity-level macro F0.5 evaluation and diagnostics."""

from dataclasses import dataclass
import argparse
import math
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from .data_loader import GROUND_TRUTH_COLUMNS, load_tsv

_S1_ID_RE = re.compile(r"^S1-[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MATCH_ID_RE = re.compile(r"^S[23]-[A-Za-z0-9][A-Za-z0-9._:-]*$")


class EvaluationInputError(ValueError):
    """Raised when truth or prediction rows are malformed or ambiguous."""


@dataclass(frozen=True)
class EvaluationReport:
    macro_f05: float
    pair_precision: float
    pair_recall: float
    singleton_accuracy: float | None
    entity_count: int
    true_pair_count: int
    predicted_pair_count: int
    country_scores: dict[str, float]


def parse_match_ids(value: object, *, reject_duplicates: bool = True) -> set[str]:
    """Parse a comma-separated S2/S3 list, validating every non-empty ID."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value)
    if not text:
        return set()
    parts = text.split(",")
    malformed = [item for item in parts if not _MATCH_ID_RE.fullmatch(item)]
    if malformed:
        raise EvaluationInputError(f"Malformed matched entity ID(s): {malformed[:5]}")
    if reject_duplicates and len(parts) != len(set(parts)):
        raise EvaluationInputError("Duplicate IDs found inside a matched_entity_ids list")
    return set(parts)


def entity_f05(true_ids: set[str], predicted_ids: set[str]) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids or not predicted_ids:
        return 0.0
    true_positives = len(true_ids & predicted_ids)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def _to_mapping(frame: pd.DataFrame, list_column: str, label: str) -> dict[str, set[str]]:
    required = {"source1_entity_id", list_column}
    missing = required - set(frame.columns)
    if missing:
        raise EvaluationInputError(f"{label} is missing column(s): {sorted(missing)}")
    duplicate_rows = frame["source1_entity_id"].duplicated(keep=False)
    if duplicate_rows.any():
        examples = frame.loc[duplicate_rows, "source1_entity_id"].astype(str).unique()[:5]
        raise EvaluationInputError(f"{label} has duplicate Source 1 rows: {list(examples)}")
    mapping: dict[str, set[str]] = {}
    for source1_id, values in frame[["source1_entity_id", list_column]].itertuples(
        index=False, name=None
    ):
        source1_id = str(source1_id)
        if not _S1_ID_RE.fullmatch(source1_id):
            raise EvaluationInputError(f"Malformed Source 1 ID in {label}: {source1_id!r}")
        mapping[source1_id] = parse_match_ids(values, reject_duplicates=True)
    return mapping


def evaluate_frames(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    countries: Mapping[str, str] | pd.DataFrame | None = None,
) -> EvaluationReport:
    """Score every truth row; an omitted prediction is treated as an empty set."""
    truth = _to_mapping(ground_truth, "matched_entity_ids", "ground truth")
    predicted = _to_mapping(predictions, "matched_entity_ids", "predictions")
    extra = set(predicted) - set(truth)
    if extra:
        raise EvaluationInputError(
            f"Predictions contain Source 1 IDs absent from ground truth: {sorted(extra)[:5]}"
        )
    if isinstance(countries, pd.DataFrame):
        if not {"entity_id", "country"}.issubset(countries.columns):
            raise EvaluationInputError("Country frame requires entity_id and country columns")
        if countries["entity_id"].duplicated().any():
            raise EvaluationInputError("Country frame contains duplicate entity_id rows")
        country_map = dict(zip(countries["entity_id"].astype(str), countries["country"].astype(str)))
    else:
        country_map = dict(countries or {})

    scores: list[float] = []
    true_pairs = predicted_pairs = true_positives = 0
    singleton_total = singleton_correct = 0
    grouped: dict[str, list[float]] = {}
    for source1_id, true_ids in truth.items():
        predicted_ids = predicted.get(source1_id, set())
        score = entity_f05(true_ids, predicted_ids)
        scores.append(score)
        true_pairs += len(true_ids)
        predicted_pairs += len(predicted_ids)
        true_positives += len(true_ids & predicted_ids)
        if not true_ids:
            singleton_total += 1
            singleton_correct += int(not predicted_ids)
        if source1_id in country_map:
            grouped.setdefault(country_map[source1_id], []).append(score)

    macro = sum(scores) / len(scores) if scores else 0.0
    pair_precision = true_positives / predicted_pairs if predicted_pairs else (1.0 if not true_pairs else 0.0)
    pair_recall = true_positives / true_pairs if true_pairs else (1.0 if not predicted_pairs else 0.0)
    return EvaluationReport(
        macro, pair_precision, pair_recall,
        singleton_correct / singleton_total if singleton_total else None,
        len(truth), true_pairs, predicted_pairs,
        {country: sum(vals) / len(vals) for country, vals in sorted(grouped.items())},
    )


def evaluate_files(
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    country_source_path: str | Path | None = None,
) -> EvaluationReport:
    truth = load_tsv(ground_truth_path, GROUND_TRUTH_COLUMNS)
    predictions = load_tsv(predictions_path, GROUND_TRUTH_COLUMNS)
    countries = load_tsv(country_source_path, ("entity_id", "country")) if country_source_path else None
    return evaluate_frames(truth, predictions, countries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate official macro F0.5")
    parser.add_argument("ground_truth")
    parser.add_argument("predictions")
    parser.add_argument("--country-source")
    args = parser.parse_args()
    report = evaluate_files(args.ground_truth, args.predictions, args.country_source)
    print(f"Macro F0.5: {report.macro_f05:.6f}")
    print(f"Pair precision: {report.pair_precision:.6f}")
    print(f"Pair recall: {report.pair_recall:.6f}")
    singleton = "n/a" if report.singleton_accuracy is None else f"{report.singleton_accuracy:.6f}"
    print(f"Singleton accuracy: {singleton}")
    for country, score in report.country_scores.items():
        print(f"Country {country}: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
