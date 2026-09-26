"""Read-only model audit by cross-script truth status, plus private error rows."""

import csv
import json
from pathlib import Path

import numpy as np

from . import config
from .evaluate import entity_f05
from .retrieval_pilot import rows, load_selected


def main():
    root = config.REPORTS_DIR / "pair_model"
    predictions = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) - {""}
                   for r in rows(root / "confirmation_matching.tsv")}
    truth_rows = load_selected(config.TRAIN_GROUND_TRUTH_PATH, set(predictions), "source1_entity_id")
    truth = {key: set(r["matched_entity_ids"].split(",")) - {""} for key, r in truth_rows.items()}
    candidates = {r["source1_entity_id"]: set(r["candidate_entity_ids"].split(",")) - {""}
                  for r in rows(root / "confirmation_candidates.tsv")}
    if set(candidates) != set(predictions) or any(predictions[key] - candidates[key] for key in predictions):
        raise ValueError("Predictions/candidate rows disagree")
    flags = {key: {"cross_script": False, "missing_address": False} for key in predictions}
    for row in rows(config.REPORTS_DIR / "retrieval_v3_confirmation50k" / "losses.tsv"):
        key = row["source1_id"]
        if key in flags:
            flags[key]["cross_script"] |= row["script_mismatch"] == "1"
            flags[key]["missing_address"] |= row["missing_address"] == "1"
    scores = {key: entity_f05(truth[key], predictions[key]) for key in predictions}
    report = {"macro_f05_independent_check": float(np.mean(list(scores.values()))), "groups": {}}
    for flag in ("cross_script", "missing_address"):
        for value in (False, True):
            selected = [key for key in predictions if flags[key][flag] == value]
            if selected:
                hits = sum(len(truth[key] & predictions[key]) for key in selected)
                predicted = sum(len(predictions[key]) for key in selected)
                actual = sum(len(truth[key]) for key in selected)
                report["groups"][f"{flag}:{value}"] = {
                    "entities": len(selected), "macro_f05": float(np.mean([scores[key] for key in selected])),
                    "pair_precision": hits / predicted if predicted else 0.,
                    "pair_recall": hits / actual if actual else 0.,
                }
    with (root / "confirmation_errors.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_id", "entity_f05", "true_count", "predicted_count", "false_positive_ids",
                         "false_negative_ids", "cross_script_truth", "missing_address_truth"])
        for key in sorted(predictions, key=lambda k: (scores[k], k)):
            if scores[key] < 1.:
                writer.writerow([key, scores[key], len(truth[key]), len(predictions[key]),
                                 ",".join(sorted(predictions[key] - truth[key])),
                                 ",".join(sorted(truth[key] - predictions[key])),
                                 int(flags[key]["cross_script"]), int(flags[key]["missing_address"])])
    expected = json.loads((root / "summary.json").read_text(encoding="utf-8"))["confirmation"]["macro_f05"]
    if abs(expected - report["macro_f05_independent_check"]) > 1e-10:
        raise ValueError("Written prediction score differs from feature-based evaluation")
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
