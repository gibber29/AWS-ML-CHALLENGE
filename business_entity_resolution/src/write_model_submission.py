"""Write test matching_results.tsv from the trained pair model.

Threshold 0.60 and the top 300 candidates match the decision that scored
macro F0.5 0.840 on the held-out 2,000 entities.
"""

from array import array
from contextlib import closing
import csv
import sqlite3
import sys

import joblib
import numpy as np

from . import config
from .pair_model import feature_vector
from .retrieval_pilot import ProjectedIndex, rows, scan_projected
from .retrieve_v2 import pack_id, unpack_id
from .retrieve_v3 import enhanced_cap, enhanced_keys, pair_view

K = 300
THRESHOLD = 0.60
MODEL = config.REPORTS_DIR / "pair_model" / "model.joblib"
POOLS = config.REPORTS_DIR / "test_pools.sqlite3"


def collect_keys():
    keys = set()
    n = 0
    print("Collecting test Source 1 keys", flush=True)
    for row in rows(config.TEST_SOURCE1_PATH):
        country = row["country"].strip()
        for key, _weight in enhanced_keys(row["business_name"], row["business_address"]):
            keys.add(country + "|" + key)
        n += 1
        if n % 200_000 == 0:
            print(f"  {n:,} entities, {len(keys):,} keys", flush=True)
    print(f"Test Source 1: {n:,}  keys: {len(keys):,}", flush=True)
    return keys


def build_index(keys):
    print("Indexing test Source 2 and Source 3", flush=True)
    index = ProjectedIndex(keys, enhanced_keys, enhanced_cap)
    scan_projected(index, (config.TEST_SOURCE2_PATH, config.TEST_SOURCE3_PATH), set(), workers=6)
    print(f"Posting keys: {len(index.postings):,}", flush=True)
    return index


def save_pools(index):
    if POOLS.exists():
        POOLS.unlink()
    wanted = set()
    n = 0
    with closing(sqlite3.connect(POOLS)) as db, db:
        db.execute(
            "CREATE TABLE pools (entity_id TEXT PRIMARY KEY, name TEXT, address TEXT, packed BLOB, score BLOB, routes TEXT)"
        )
        batch = []
        for row in rows(config.TEST_SOURCE1_PATH):
            ranked, scores, routes = index.rank(row)
            pool = ranked[:K]
            wanted.update(pool)
            best = scores[pool[0]] if pool else 1.0
            score_blob = array("d", (scores[packed] / max(best, 1e-9) for packed in pool)).tobytes()
            route_text = ";".join(",".join(sorted(routes[packed])) for packed in pool)
            batch.append((
                row["entity_id"], row["business_name"], row["business_address"],
                array("Q", pool).tobytes(), score_blob, route_text,
            ))
            n += 1
            if len(batch) >= 1000:
                db.executemany("INSERT INTO pools VALUES (?,?,?,?,?,?)", batch)
                batch.clear()
            if n % 20_000 == 0:
                print(f"  pooled {n:,}  unique candidates {len(wanted):,}", flush=True)
        if batch:
            db.executemany("INSERT INTO pools VALUES (?,?,?,?,?,?)", batch)
    print(f"Pooled {n:,} entities, {len(wanted):,} unique candidates", flush=True)
    return wanted


def load_records(wanted):
    found = {}
    for path in (config.TEST_SOURCE2_PATH, config.TEST_SOURCE3_PATH):
        kept = 0
        for i, row in enumerate(rows(path), 1):
            packed = pack_id(row["entity_id"])
            if packed in wanted:
                found[packed] = (row["business_name"], row["business_address"])
                kept += 1
            if i % 1_000_000 == 0:
                print(f"  {path.name}: {i:,}  kept {kept:,}", flush=True)
    print(f"Candidate records loaded: {len(found):,}", flush=True)
    return found


def write_matches(model, records):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = config.OUTPUT_DIR / "matching_results.tsv"
    written = 0
    nonempty = 0
    pending = []
    vectors = []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        def flush():
            nonlocal written, nonempty
            if not pending:
                return
            probabilities = model.predict_proba(np.asarray(vectors, dtype=np.float64))[:, 1]
            cursor = 0
            for entity_id, pool in pending:
                count = len(pool)
                chosen = [
                    unpack_id(packed)
                    for packed, probability in zip(pool, probabilities[cursor:cursor + count])
                    if probability >= THRESHOLD
                ]
                cursor += count
                writer.writerow([entity_id, ",".join(chosen)])
                written += 1
                nonempty += bool(chosen)
            pending.clear()
            vectors.clear()

        with closing(sqlite3.connect(POOLS)) as db:
            for n, (entity_id, name, address, packed_blob, score_blob, route_text) in enumerate(
                db.execute("SELECT * FROM pools"), 1
            ):
                pool = array("Q")
                pool.frombytes(packed_blob)
                rel = array("d")
                rel.frombytes(score_blob)
                route_sets = [set(filter(None, part.split(","))) for part in route_text.split(";")] if pool else []
                left = pair_view(name, address)
                usable = []
                for packed, relative, route_set in zip(pool, rel, route_sets):
                    record = records.get(int(packed))
                    if record is None:
                        continue
                    vectors.append(feature_vector(left, pair_view(*record), float(relative), route_set, int(packed)))
                    usable.append(int(packed))
                pending.append((entity_id, usable))
                if len(pending) >= 128:
                    flush()
                if n % 20_000 == 0:
                    print(f"  scored {n:,}", flush=True)
            flush()
    print(f"Rows: {written:,}  with matches: {nonempty:,}", flush=True)
    print(f"Saved {output}", flush=True)


def main() -> int:
    if not MODEL.exists():
        raise SystemExit(f"Missing model: {MODEL}")
    print("Collecting keys", flush=True)
    keys = collect_keys()
    index = build_index(keys)
    wanted = save_pools(index)
    del index
    records = load_records(wanted)
    write_matches(joblib.load(MODEL), records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
