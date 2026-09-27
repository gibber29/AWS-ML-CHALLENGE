"""Score a completed test_pools.sqlite3 without rebuilding the retrieval index.

Place this file in the AWS-ML-CHALLENGE repository root. Run only after
write_model_submission.py has FINISHED its pooling phase and the SQLite file
contains every test Source 1 row. Choose --k after running quick_k_sweep.py.

    py -3.11 score_saved_pools.py --k 50

Outputs are placed in output/model_k50/ so the prior submission is preserved.
The candidate file contains exactly the candidates passed to the model.
"""

import argparse
from array import array
from contextlib import closing
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import joblib
import numpy as np

from business_entity_resolution.src import config
from business_entity_resolution.src.pair_model import FEATURES, feature_vector
from business_entity_resolution.src.retrieval_pilot import rows
from business_entity_resolution.src.retrieve_v2 import pack_id, unpack_id
from business_entity_resolution.src.retrieve_v3 import pair_view


def read_pool(row, k):
    entity_id, name, address, packed_blob, score_blob, route_text = row
    packed = array("Q")
    packed.frombytes(packed_blob)
    scores = array("d")
    scores.frombytes(score_blob)
    route_parts = route_text.split(";") if packed else []
    if len(packed) != len(scores) or len(packed) != len(route_parts):
        raise ValueError(f"Corrupt candidate pool for {entity_id}")
    return entity_id, name, address, list(packed[:k]), list(scores[:k]), route_parts[:k]


def collect_wanted(db, k):
    wanted = set()
    for n, row in enumerate(db.execute("SELECT * FROM pools ORDER BY rowid"), 1):
        _id, _name, _address, pool, _scores, _routes = read_pool(row, k)
        wanted.update(pool)
        if n % 200_000 == 0:
            print(f"  selected {n:,} S1 pools; {len(wanted):,} unique candidate IDs", flush=True)
    return wanted


def load_records(wanted):
    found = {}
    for path in (config.TEST_SOURCE2_PATH, config.TEST_SOURCE3_PATH):
        for n, row in enumerate(rows(path), 1):
            packed = pack_id(row["entity_id"])
            if packed in wanted:
                found[packed] = (row["business_name"], row["business_address"])
            if n % 1_000_000 == 0:
                print(f"  scanned {path.name}: {n:,}", flush=True)
    if len(found) != len(wanted):
        raise ValueError(f"Missing {len(wanted) - len(found):,} selected test candidate records")
    print(f"Loaded {len(found):,} unique candidate records", flush=True)
    return found


def score(db, records, model, threshold, k, output, expected_rows):
    output.mkdir(parents=True, exist_ok=True)
    matching = output / "matching_results.tsv"
    candidates = output / "candidate_pairs.tsv"
    matching_tmp = output / "matching_results.tsv.tmp"
    candidates_tmp = output / "candidate_pairs.tsv.tmp"
    if matching.exists() or candidates.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result in {output}")
    if matching_tmp.exists() or candidates_tmp.exists():
        raise FileExistsError("Temporary output exists; inspect/remove it before retrying")

    @lru_cache(maxsize=50_000)
    def candidate_view(packed):
        return pair_view(*records[packed])

    pending = []
    vectors = []
    written = 0
    nonempty = 0

    with matching_tmp.open("w", encoding="utf-8", newline="") as mh, \
            candidates_tmp.open("w", encoding="utf-8", newline="") as ch:
        match_writer = csv.writer(mh, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(ch, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        def flush():
            nonlocal written, nonempty
            if not pending:
                return
            if vectors:
                probabilities = model.predict_proba(np.asarray(vectors, dtype=np.float32))[:, 1]
            else:
                probabilities = np.empty(0, dtype=np.float32)
            cursor = 0
            for entity_id, pool in pending:
                size = len(pool)
                accepted = [unpack_id(p) for p, probability in
                            zip(pool, probabilities[cursor:cursor + size])
                            if probability >= threshold]
                cursor += size
                match_writer.writerow([entity_id, ",".join(accepted)])
                candidate_writer.writerow([entity_id, ",".join(unpack_id(p) for p in pool)])
                written += 1
                nonempty += bool(accepted)
            pending.clear()
            vectors.clear()

        for n, row in enumerate(db.execute("SELECT * FROM pools ORDER BY rowid"), 1):
            entity_id, name, address, pool, scores, routes = read_pool(row, k)
            left = pair_view(name, address)
            for packed, relative, route_text in zip(pool, scores, routes):
                vectors.append(feature_vector(left, candidate_view(packed), float(relative),
                                              set(filter(None, route_text.split(","))), packed))
            pending.append((entity_id, pool))
            if len(pending) >= 128:
                flush()
            if n % 20_000 == 0:
                print(f"  scored {n:,} S1; {nonempty:,} nonempty", flush=True)
        flush()
    if written != expected_rows:
        raise ValueError(f"Wrote {written:,} rows; expected {expected_rows:,}. Temporary files retained.")
    os.replace(matching_tmp, matching)
    os.replace(candidates_tmp, candidates)
    print(f"Completed {written:,} rows; {nonempty:,} with a match")
    print(f"Matching: {matching}\nCandidates: {candidates}")
    print("Now run the official validator. Upload ONLY matching_results.tsv to the portal.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pool-db", type=Path, default=config.REPORTS_DIR / "test_pools.sqlite3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.k <= 300:
        raise SystemExit("--k must be between 1 and the saved pool size 300")
    output = args.output or config.OUTPUT_DIR / f"model_k{args.k}"
    model_path = config.REPORTS_DIR / "pair_model" / "model.joblib"
    summary_path = config.REPORTS_DIR / "pair_model" / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != report["model_sha256"]:
        raise ValueError("Model checksum differs from the training report")
    if tuple(report["features"]) != FEATURES:
        raise ValueError("Feature order differs from the trained model")
    model = joblib.load(model_path)
    if model.n_features_in_ != len(FEATURES):
        raise ValueError("Model feature count differs from scorer")
    threshold = float(report["threshold_selection"]["threshold"])
    with closing(sqlite3.connect(f"{args.pool_db.resolve().as_uri()}?mode=ro", uri=True)) as db:
        pooled = db.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
        expected_rows = sum(1 for _ in rows(config.TEST_SOURCE1_PATH))
        if pooled != expected_rows:
            raise ValueError(f"Incomplete pools: {pooled:,} versus {expected_rows:,} test S1 records")
        print(f"Using {pooled:,} completed pools at K={args.k}, threshold={threshold:.3f}", flush=True)
        wanted = collect_wanted(db, args.k)
        records = load_records(wanted)
        score(db, records, model, threshold, args.k, output, expected_rows)


if __name__ == "__main__":
    main()
