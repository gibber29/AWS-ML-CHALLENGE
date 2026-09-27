"""Compare inference cutoffs using the existing confirmation feature cache.

Run from AWS-ML-CHALLENGE repository root:
    py -3.11 quick_k_sweep.py

This reads only existing model/validation artifacts. It neither retrains nor
changes the saved submission files.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def score_cutoff(labels, probabilities, truth_counts, offsets, countries, cutoff, threshold):
    totals = {"score": 0.0, "oracle": 0.0, "tp": 0, "predicted": 0,
              "true": 0, "singleton_correct": 0, "singletons": 0}
    by_country = {}
    for i, truth_count in enumerate(truth_counts):
        start, end = int(offsets[i]), int(offsets[i + 1])
        stop = min(end, start + cutoff)
        selected = probabilities[start:stop] >= threshold
        tp = int(labels[start:stop][selected].sum())
        predicted = int(selected.sum())
        available = int(labels[start:stop].sum())
        truth_count = int(truth_count)
        score = (1.25 * tp / (predicted + 0.25 * truth_count)
                 if predicted or truth_count else 1.0)
        oracle = (1.25 * available / (available + 0.25 * truth_count)
                  if truth_count else 1.0)
        totals["score"] += score
        totals["oracle"] += oracle
        totals["tp"] += tp
        totals["predicted"] += predicted
        totals["true"] += truth_count
        if truth_count == 0:
            totals["singletons"] += 1
            totals["singleton_correct"] += int(predicted == 0)
        country = str(countries[i])
        entry = by_country.setdefault(country, [0.0, 0])
        entry[0] += score
        entry[1] += 1
    n = len(truth_counts)
    return {
        "k": cutoff, "threshold": threshold, "entities": n,
        "macro_f05": totals["score"] / n,
        "oracle_macro_f05": totals["oracle"] / n,
        "pair_precision": totals["tp"] / totals["predicted"] if totals["predicted"] else 0.0,
        "pair_recall": totals["tp"] / totals["true"] if totals["true"] else 0.0,
        "singleton_accuracy": (totals["singleton_correct"] / totals["singletons"]
                               if totals["singletons"] else None),
        "predicted_pairs": totals["predicted"],
        "country_f05": {country: value / count for country, (value, count) in by_country.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/pair_model"))
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[20, 50, 100, 150, 300])
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    threshold = float(report["threshold_selection"]["threshold"])
    cache = args.root / "confirm_features" / "features.npz"
    if not cache.exists():
        raise SystemExit(f"Missing confirmation cache: {cache}")
    with np.load(cache, allow_pickle=False) as data:
        x = data["x"]
        labels = data["y"]
        truth_counts = data["truth_counts"]
        offsets = data["offsets"]
        countries = data["countries"]
    if len(x) != len(labels) or len(offsets) != len(truth_counts) + 1:
        raise ValueError("Confirmation feature cache has inconsistent dimensions")
    if int(offsets[-1]) != len(x):
        raise ValueError("Confirmation offsets do not cover the feature rows")
    model = joblib.load(args.root / "model.joblib")
    probabilities = model.predict_proba(x)[:, 1]
    results = [score_cutoff(labels, probabilities, truth_counts, offsets, countries,
                            k, threshold) for k in args.cutoffs]
    at_300 = next((row for row in results if row["k"] == 300), None)
    expected = report["confirmation"]["macro_f05"]
    if at_300 and abs(at_300["macro_f05"] - expected) > 1e-8:
        raise ValueError(f"K=300 score {at_300['macro_f05']:.8f} differs from saved {expected:.8f}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
