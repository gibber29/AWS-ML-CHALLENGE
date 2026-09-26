"""Compare additive retrieval and name/address reranking on an existing pilot."""

import argparse
from array import array
from collections import defaultdict
from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

from . import config
from .evaluate import entity_f05
from .retrieval_pilot import ProjectedIndex, Timer, rows, load_selected, digest_files, measure, KS, scan_projected
from .retrieve_v2 import pack_id
from .retrieve_v3 import (VERSION, enhanced_keys, enhanced_cap, pair_view,
                          pair_features, pair_rank, route_pool)


def candidate_records(path, signature, wanted):
    if path.exists():
        with closing(sqlite3.connect(path)) as db:
            saved = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            if saved != signature:
                raise ValueError("Candidate-record cache differs; use a new output directory")
        return
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE metadata (value TEXT)")
        db.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, name TEXT, address TEXT)")
        batch = []
        found = 0
        for source in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
            for i, row in enumerate(rows(source), 1):
                packed = pack_id(row["entity_id"])
                if packed in wanted:
                    batch.append((packed, row["business_name"], row["business_address"]))
                    found += 1
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO records VALUES (?,?,?)", batch)
                    batch.clear()
                if i % 1000000 == 0:
                    print(f"materialize {source.name}: {i:,}; selected {found:,}", flush=True)
        db.executemany("INSERT INTO records VALUES (?,?,?)", batch)
        if found != len(wanted):
            raise ValueError("Some pool candidate IDs are missing")
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(signature),))


def rerank_metrics(index, source1, truth, output, signature, pool_size):
    pools, wanted = {}, set()
    for i, (entity_id, row) in enumerate(source1.items(), 1):
        ranked, _, routes = index.rank(row)
        pool = route_pool(ranked, routes, pool_size)
        pools[entity_id] = array("Q", pool)
        wanted.update(pool)
        if i % 250 == 0:
            print(f"prepare pools: {i:,}", flush=True)
    pool_hash = hashlib.sha256(b"".join(k.encode() + v.tobytes() for k, v in sorted(pools.items()))).hexdigest()
    path = output / "pool_records.sqlite3"
    candidate_records(path, {"index": signature, "pool_size": pool_size, "pool_hash": pool_hash}, wanted)
    del wanted
    report = defaultdict(lambda: {"entities": 0, "true_links": 0, "at_k": {
        str(k): {"hits": 0, "oracle_sum": 0., "covered": 0} for k in (*KS, "pool")}})

    @lru_cache(maxsize=20000)
    def view(name, address):
        return pair_view(name, address)

    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        for i, (entity_id, row) in enumerate(source1.items(), 1):
            pool = pools[entity_id]
            left = pair_view(row["business_name"], row["business_address"])
            features = {}
            for offset in range(0, len(pool), 900):
                ids = list(pool[offset:offset+900])
                sql = "SELECT * FROM records WHERE id IN (" + ",".join("?" for _ in ids) + ")"
                for candidate, name, address in db.execute(sql, ids):
                    features[candidate] = pair_features(left, view(name, address))
            orders = {"route_pool_idf": list(pool)}
            for mode, address in (("name_only", False), ("name_address", True)):
                orders[mode] = sorted(pool, key=lambda p: (-pair_rank(features[p], address), p))
            labels = {pack_id(label) for label in truth[entity_id]}
            native = any(ord(c) > 127 and c.isalpha() for c in row["business_name"])
            for mode, order in orders.items():
                for group in ("all", "country:" + row["country"], "native_name:" + str(native)):
                    totals = report[mode + "/" + group]
                    totals["entities"] += 1
                    totals["true_links"] += len(labels)
                    for k in (*KS, "pool"):
                        hits = labels & set(order if k == "pool" else order[:k])
                        totals["at_k"][str(k)]["hits"] += len(hits)
                        totals["at_k"][str(k)]["oracle_sum"] += entity_f05(labels, hits)
                        totals["at_k"][str(k)]["covered"] += bool(hits)
            if i % 250 == 0:
                print(f"reranked: {i:,}", flush=True)
    for group in report.values():
        for bucket in group["at_k"].values():
            bucket["pair_recall"] = bucket["hits"] / group["true_links"] if group["true_links"] else None
            bucket["oracle_macro_f05"] = bucket.pop("oracle_sum") / group["entities"]
    return dict(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=config.REPORTS_DIR / "retrieval_pilot")
    parser.add_argument("--output", type=Path, default=config.REPORTS_DIR / "retrieval_v3_pilot")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4, help="Local CPU key-extraction workers")
    parser.add_argument("--skip-rerank", action="store_true", help="Measure retrieval only (e.g. 50k confirmation)")
    parser.add_argument("--rerank-baseline", action="store_true", help="Reuse dedup baseline index without adding keys")
    args = parser.parse_args()
    if args.pool_size < max(KS):
        parser.error("--pool-size must be >=1000 for the reported K values")
    timer = Timer()
    ids = {r["source1_entity_id"] for r in rows(args.pilot_dir / "pilot_ids.tsv")}
    source1 = dict(sorted(load_selected(config.TRAIN_SOURCE1_PATH, ids).items()))
    truth_rows = load_selected(config.TRAIN_GROUND_TRUTH_PATH, ids, "source1_entity_id")
    truth = {key: set(r["matched_entity_ids"].split(",")) - {""} for key, r in truth_rows.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "pilot_size": len(ids), "pool_size": args.pool_size}
    if args.rerank_baseline:
        baseline = json.loads((args.pilot_dir / "summary.json").read_text(encoding="utf-8"))
        signature = baseline["signature"]
        # Reject changed inputs even when the caller passes a historical report.
        for stamp in signature["files"]:
            if digest_files([Path(stamp["path"])])[0] != stamp:
                raise ValueError("Baseline input changed; rerun pilot first")
        index, targets = ProjectedIndex.load(args.pilot_dir / "projected_index.sqlite3", signature)
        report["version"] = "dedup-baseline-rerank"
        timer.mark("load_baseline")
    else:
        keys = {row["country"].strip() + "|" + key for row in source1.values()
                for key, _ in enhanced_keys(row["business_name"], row["business_address"])}
        signature = {"version": VERSION, "files": digest_files([
            config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH,
            config.TRAIN_GROUND_TRUTH_PATH, args.pilot_dir / "pilot_ids.tsv"]),
            "key_hash": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
            "code_hash": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                for name in ("retrieve_v3.py", "retrieve_v2.py", "normalize.py", "block.py"))).hexdigest()}
        cache = args.output / "projected_index.sqlite3"
        timer.mark("load_pilot")
        if cache.exists():
            index, targets = ProjectedIndex.load(cache, signature, enhanced_keys)
            timer.mark("load_cache")
        else:
            index = ProjectedIndex(keys, enhanced_keys, enhanced_cap)
            true_ids = set().union(*truth.values())
            targets = scan_projected(index, (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH),
                                     true_ids, args.workers)
            if set(targets) != true_ids:
                raise ValueError("Missing true candidate records")
            timer.mark("count_and_bounded_postings")
            timer.phases["count_and_bounded_postings"].update(index.scan_profile)
            index.save(cache, signature, targets)
            timer.mark("save_cache")
        report["retrieval"] = measure(index, source1, truth, targets, args.output / "losses.tsv")
        timer.mark("query_enhanced")
        print(json.dumps(report["retrieval"]["groups"]["all"], indent=2), flush=True)
        # Save measured retrieval before the longer materialization/reranking pass.
        (args.output / "retrieval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["signature"] = signature
    if args.skip_rerank:
        report["phases"] = timer.phases
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return
    report["rerank"] = rerank_metrics(index, source1, truth, args.output, signature, args.pool_size)
    timer.mark("pool_materialization_and_rerank")
    report["phases"] = timer.phases
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for mode in ("route_pool_idf", "name_only", "name_address"):
        print(mode, json.dumps(report["rerank"][mode + "/all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
