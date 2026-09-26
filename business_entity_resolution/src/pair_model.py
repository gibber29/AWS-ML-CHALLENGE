"""Train a compact pair decision on retrieved hard negatives, with entity metrics.

No test data, external entity lookup, pretrained model, IDs, country labels, or
truth match counts are model features. Truth counts are used only for evaluation.
"""

import argparse
from collections import defaultdict
from contextlib import closing
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
import sys

from . import config
from .retrieval_pilot import ProjectedIndex, rows, load_selected, digest_files, Timer
from .retrieval_experiment import candidate_records
from .retrieve_v2 import pack_id, unpack_id
from .retrieve_v3 import enhanced_keys, pair_view, pair_features
from .split import SPLIT_PATH

FEATURES = ("name_exact", "name_tokens", "name_chars", "name_skeleton", "address_words",
            "address_containment", "number_long", "number_medium", "number_short",
            "number_conflict", "missing_address", "name_best", "name_address_interaction",
            "skeleton_address_interaction", "number_name_interaction", "name_length_ratio",
            "retrieval_relative_score", "route_count", "source3")


def prepare_membership(output, partition, size, exclude):
    groups = defaultdict(list)
    for row in rows(SPLIT_PATH):
        eligible = row["split"] == partition and (partition == "train" or row["dev_sample"] == "1")
        if eligible and row["source1_entity_id"] not in exclude:
            groups[row["country"], row["bucket"]].append(row["source1_entity_id"])
    total = sum(map(len, groups.values()))
    if not 0 < size <= total:
        raise ValueError("Sample size exceeds eligible partition")
    quotas = {key: int(size * len(ids) / total) for key, ids in groups.items()}
    while sum(quotas.values()) < size:
        key = max(groups, key=lambda k: (size * len(groups[k]) / total - quotas[k], k))
        quotas[key] += 1
    rng, selected = random.Random(config.RANDOM_SEED), []
    for key, ids in sorted(groups.items()):
        ids.sort()
        rng.shuffle(ids)
        selected.extend(ids[:quotas[key]])
    source = load_selected(config.TRAIN_SOURCE1_PATH, set(selected))
    truth = load_selected(config.TRAIN_GROUND_TRUTH_PATH, set(selected), "source1_entity_id")
    output.mkdir(parents=True, exist_ok=True)
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["source1_entity_id", "country", "match_count"])
    writer.writerows((key, source[key]["country"], len(set(truth[key]["matched_entity_ids"].split(",")) - {""}))
                     for key in sorted(selected))
    path = output / "pilot_ids.tsv"
    if not path.exists() or path.read_text(encoding="utf-8") != handle.getvalue():
        path.write_text(handle.getvalue(), encoding="utf-8", newline="")
    print(f"Prepared {size} {partition} S1 entities: {output}", flush=True)


def load_index(path):
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        signature = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])["signature"]
    for stamp in signature["files"]:
        if digest_files([Path(stamp["path"])])[0] != stamp:
            raise ValueError("Index input changed")
    code_hash = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
        for name in ("retrieve_v3.py", "retrieve_v2.py", "normalize.py", "block.py"))).hexdigest()
    if signature["code_hash"] != code_hash:
        raise ValueError("Index key/normalization code changed")
    return (*ProjectedIndex.load(path, signature, enhanced_keys), signature)


def feature_vector(left, right, relative_score, routes, packed):
    f = pair_features(left, right)
    name = max(f["name_tokens"], f["name_chars"], f["name_skeleton"])
    independent = (bool(routes & {"t", "c", "f_t", "f_c"}),
                   bool(routes & {"p", "s", "sk", "sg"}),
                   bool(routes & {"aw", "f_aw", "a2"}),
                   bool(routes & {"n", "ns", "n2", "an"}))
    f.update(name_best=name, name_address_interaction=name * f["address_words"],
             skeleton_address_interaction=f["name_skeleton"] * f["address_words"],
             number_name_interaction=name * f["number_short"],
             name_length_ratio=min(len(left.name), len(right.name)) / max(1, len(left.name), len(right.name)),
             retrieval_relative_score=relative_score, route_count=sum(independent),
             source3=float(packed >> 40 == 1))
    return [f[key] for key in FEATURES]


def extract(index, signature, manifest, output, k):
    import numpy as np
    ids = sorted(r["source1_entity_id"] for r in rows(manifest / "pilot_ids.tsv"))
    output.mkdir(parents=True, exist_ok=True)
    schema = {"index": signature, "ids_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
              "k": k, "features": FEATURES, "feature_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    schema = json.loads(json.dumps(schema))
    cache = output / "features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["schema"])) != schema:
                raise ValueError("Feature cache changed; choose a new output directory")
            return {key: saved[key] for key in saved.files if key != "schema"}
    source = load_selected(config.TRAIN_SOURCE1_PATH, set(ids))
    truths = load_selected(config.TRAIN_GROUND_TRUTH_PATH, set(ids), "source1_entity_id")
    truth = {key: set(row["matched_entity_ids"].split(",")) - {""} for key, row in truths.items()}
    pools, scores_by_query, routes_by_query, wanted = {}, {}, {}, set()
    for key in ids:
        row = source[key]
        if any(row["country"].strip() + "|" + k not in index.df
               for k, _ in enhanced_keys(row["business_name"], row["business_address"])):
            raise ValueError("Selected S1 requires keys missing from this index")
        ranked, scores, routes = index.rank(row)
        pool = ranked[:k]
        pools[key] = pool
        scores_by_query[key] = {p: scores[p] / max(1e-9, scores[ranked[0]]) for p in pool}
        routes_by_query[key] = {p: routes[p] for p in pool}
        wanted.update(pool)
    records_path = output / "records.sqlite3"
    candidate_records(records_path, schema, wanted)
    del wanted
    count = sum(map(len, pools.values()))
    x, y, candidates = np.empty((count, len(FEATURES)), np.float32), np.zeros(count, np.uint8), np.empty(count, np.int64)
    offsets, truth_counts, countries = [0], [], []
    position = 0

    @lru_cache(maxsize=10000)
    def view(name, address):
        return pair_view(name, address)

    with closing(sqlite3.connect(f"{records_path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        for i, key in enumerate(ids, 1):
            row, pool = source[key], pools[key]
            left = pair_view(row["business_name"], row["business_address"])
            records = {}
            for start in range(0, len(pool), 900):
                part = pool[start:start+900]
                records.update({p: (name, address) for p, name, address in db.execute(
                    "SELECT * FROM records WHERE id IN (" + ",".join("?" for _ in part) + ")", part)})
            labels = {pack_id(label) for label in truth[key]}
            for candidate in pool:
                x[position] = feature_vector(left, view(*records[candidate]), scores_by_query[key][candidate],
                                             routes_by_query[key][candidate], candidate)
                y[position], candidates[position] = candidate in labels, candidate
                position += 1
            offsets.append(position)
            truth_counts.append(len(labels))
            countries.append(row["country"])
            if i % 250 == 0:
                print(f"features {output.name}: {i:,}", flush=True)
    result = dict(x=x, y=y, candidates=candidates, offsets=np.array(offsets), truth_counts=np.array(truth_counts),
                  ids=np.array(ids), countries=np.array(countries))
    np.savez_compressed(cache, schema=json.dumps(schema), **result)
    return result


def evaluate(data, probabilities, threshold):
    import numpy as np
    scores, oracle, hits, predicted, singleton_correct, complete = [], [], 0, 0, 0, 0
    country_scores = defaultdict(list)
    predicted_counts = []
    for i, total in enumerate(data["truth_counts"]):
        start, end = data["offsets"][i:i+2]
        labels = data["y"][start:end]
        accepted = probabilities[start:end] >= threshold
        tp, n = int(labels[accepted].sum()), int(accepted.sum())
        available = int(labels.sum())
        score = 1.25 * tp / (n + .25 * total) if total or n else 1.
        ceiling = 1.25 * available / (available + .25 * total) if total else 1.
        scores.append(score)
        oracle.append(ceiling)
        country_scores[str(data["countries"][i])].append(score)
        hits += tp
        predicted += n
        singleton_correct += int(total == 0 and n == 0)
        complete += int(total > 0 and tp == total)
        predicted_counts.append(n)
    true_pairs = int(data["truth_counts"].sum())
    singletons = int((data["truth_counts"] == 0).sum())
    return {"threshold": float(threshold), "entities": len(scores), "macro_f05": float(np.mean(scores)),
            "oracle_macro_f05": float(np.mean(oracle)), "gap_to_oracle": float(np.mean(oracle) - np.mean(scores)),
            "pair_precision": hits / predicted if predicted else (0. if true_pairs else 1.),
            "pair_recall": hits / true_pairs if true_pairs else (0. if predicted else 1.),
            "singleton_accuracy": singleton_correct / singletons if singletons else None,
            "all_match_completeness": complete / (len(scores) - singletons) if len(scores) != singletons else None,
            "true_pairs": true_pairs, "predicted_pairs": predicted, "true_positives": hits,
            "predicted_empty_rate": sum(n == 0 for n in predicted_counts) / len(scores),
            "mean_predicted_matches": float(np.mean(predicted_counts)), "mean_true_matches": float(np.mean(data["truth_counts"])),
            "country_f05": {key: float(np.mean(values)) for key, values in country_scores.items()}}


def write_predictions(data, probabilities, threshold, output):
    """Validation artifacts include every S1, including empty candidate lists."""
    with (output / "confirmation_matching.tsv").open("w", encoding="utf-8", newline="") as matches, \
            (output / "confirmation_candidates.tsv").open("w", encoding="utf-8", newline="") as candidates:
        match_writer, candidate_writer = csv.writer(matches, delimiter="\t"), csv.writer(candidates, delimiter="\t")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for i, entity_id in enumerate(data["ids"]):
            start, end = data["offsets"][i:i+2]
            pool = data["candidates"][start:end]
            predicted = pool[probabilities[start:end] >= threshold]
            match_writer.writerow([entity_id, ",".join(unpack_id(int(p)) for p in predicted)])
            candidate_writer.writerow([entity_id, ",".join(unpack_id(int(p)) for p in pool)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "features", "fit"))
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--root", type=Path, default=config.REPORTS_DIR / "pair_model")
    args = parser.parse_args()
    pilot = config.REPORTS_DIR / "retrieval_pilot"
    if args.action == "prepare":
        exclude = {r["source1_entity_id"] for r in rows(pilot / "pilot_ids.tsv")}
        prepare_membership(args.root / "train", "train", args.size, set())
        prepare_membership(args.root / "confirmation", "valid", args.size, exclude)
        return
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    timer = Timer()
    if args.action == "fit":
        retrieval_confirmation = json.loads((config.REPORTS_DIR / "retrieval_v3_confirmation50k" / "summary.json").read_text(encoding="utf-8"))
        ceiling = retrieval_confirmation["retrieval"]["groups"]["all"]["at_k"][str(args.k)]["oracle_macro_f05"]
        if ceiling < .98:
            raise ValueError(f"Retrieval gate not met at K={args.k}: confirmation oracle={ceiling:.6f}")
    train_index, train_targets, train_signature = load_index(args.root / "train_index" / "projected_index.sqlite3")
    valid_index, valid_targets, valid_signature = load_index(config.REPORTS_DIR / "retrieval_v3_confirmation50k" / "projected_index.sqlite3")
    overlap = set(train_targets) & set(valid_targets)
    if overlap:
        raise ValueError(f"S2/S3 label leakage between training and validation: {len(overlap)} IDs")
    train = extract(train_index, train_signature, args.root / "train", args.root / "train_features", args.k)
    tune = extract(valid_index, valid_signature, pilot, args.root / "tune_features", args.k)
    confirm = extract(valid_index, valid_signature, args.root / "confirmation", args.root / "confirm_features", args.k)
    del train_index, valid_index, train_targets, valid_targets
    if set(train["ids"]) & (set(tune["ids"]) | set(confirm["ids"])) or set(tune["ids"]) & set(confirm["ids"]):
        raise ValueError("Entity partitions overlap")
    train_ids, valid_ids = set(train["ids"]), set(tune["ids"]) | set(confirm["ids"])
    selected_ids = train_ids | valid_ids
    verified = set()
    for row in rows(SPLIT_PATH):
        key = row["source1_entity_id"]
        if key in selected_ids:
            if row["split"] != ("train" if key in train_ids else "valid"):
                raise ValueError("Entity violates frozen training/validation assignment")
            verified.add(key)
    if verified != selected_ids:
        raise ValueError("Missing frozen split assignments")
    timer.mark("extract_features")
    if args.action == "features":
        preparation = {"phases": timer.phases, "k": args.k,
                       "pairs": {"train": len(train["y"]), "tune": len(tune["y"]), "confirm": len(confirm["y"])}}
        (args.root / "feature_preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
        print("Cached train, threshold-selection, and confirmation pair features; no classifier fitted.", flush=True)
        return
    # Equal total weight per S1, including every retrieved hard negative.
    lengths = np.diff(train["offsets"])
    weights = np.repeat(1. / np.maximum(1, lengths), lengths)
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.08,
                                          l2_regularization=1., early_stopping=False, random_state=2026)
    with threadpool_limits(limits=4):
        model.fit(train["x"], train["y"], sample_weight=weights)
        tune_prob = model.predict_proba(tune["x"])[:, 1]
        confirm_prob = model.predict_proba(confirm["x"])[:, 1]
    timer.mark("fit_and_predict")
    sweep = [evaluate(tune, tune_prob, threshold) for threshold in np.arange(.05, .96, .025)]
    best = max(sweep, key=lambda r: (r["macro_f05"], r["threshold"]))
    report = {"features": FEATURES, "k": args.k, "training_entities": len(train["ids"]),
              "training_pairs": len(train["y"]), "training_positive_pairs": int(train["y"].sum()),
              "training_validation_shared_truth_ids": len(overlap),
              "sklearn_version": sklearn.__version__, "sklearn_library_license": "BSD-3-Clause",
              "numpy_version": np.__version__, "python_version": sys.version,
              "model_parameters": model.get_params(), "train_index_signature": train_signature,
              "valid_index_signature": valid_signature,
              "pretrained_model": False, "threshold_selection": best,
              "confirmation": evaluate(confirm, confirm_prob, best["threshold"]), "threshold_sweep": sweep,
              "note": "Compact model trained from challenge data; no external pretrained weights. Pilot used for threshold selection. Confirmation entities disjoint from pilot and train. No test inference.",
              "phases": timer.phases}
    joblib.dump(model, args.root / "model.joblib")
    write_predictions(confirm, confirm_prob, best["threshold"], args.root)
    report["model_sha256"] = hashlib.sha256((args.root / "model.joblib").read_bytes()).hexdigest()
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"threshold_selection": best, "confirmation": report["confirmation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
