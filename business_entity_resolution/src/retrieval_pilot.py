"""Exact, query-projected retrieval diagnosis against the entire training corpus.

Only keys emitted by the frozen pilot are retained. Counts still include every
S2/S3 row, so caps, IDF and ranks match a full index. Truth never selects keys or
adds candidates. The legacy and document-frequency variants share one scan.
"""

import argparse
from array import array
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import unicodedata

from . import config
from .block import _core_tokens
from .evaluate import entity_f05
from .retrieve_v2 import _cap, legacy_record_keys, record_keys, pack_id, unique_keys
from .split import SPLIT_PATH, build_split, match_bucket

KS = (20, 50, 100, 300, 1000)
VERSION = "projected-v1"
REASONS = ("NO_RAW_KEY", "ALL_SHARED_KEYS_PRUNED", "RANKED_BELOW_K", "IN_TOP_K")


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def scripts(name):
    return frozenset(unicodedata.name(c, "UNKNOWN").split()[0]
                     for c in name if c.isalpha())


def rss_mb():
    """Actual process working set and lifetime peak, not Python traced allocation."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in (
                    "peak", "working", "paged_peak", "paged", "nonpaged_peak",
                    "nonpaged", "pagefile", "pagefile_peak")]
        value = Memory()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_mb": value.working / 2**20, "process_peak_rss_mb": value.peak / 2**20}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_mb": peak / (2**20 if sys.platform == "darwin" else 1024)}


class Timer:
    def __init__(self):
        self.phases = {}
        self.start = time.perf_counter()

    def mark(self, phase):
        now = time.perf_counter()
        self.phases[phase] = {"seconds": now - self.start, **rss_mb()}
        self.start = now
        print(f"{phase}: {json.dumps(self.phases[phase])}", flush=True)


def load_selected(path, ids, id_column="entity_id"):
    found = {}
    for row in rows(path):
        key = row[id_column]
        if key in ids:
            if key in found:
                raise ValueError(f"Duplicate ID: {key}")
            found[key] = row
    if set(found) != set(ids):
        raise ValueError(f"Missing {len(set(ids) - set(found))} records from {path}")
    return found


def select_pilot(source1, truth, size, seed):
    if not 0 < size <= len(source1):
        raise ValueError(f"Pilot size must be between 1 and {len(source1)}")
    names = Counter((r["country"], " ".join(_core_tokens(r["business_name"])))
                    for r in source1.values())
    groups = defaultdict(list)
    for entity_id, row in source1.items():
        core = " ".join(_core_tokens(row["business_name"]))
        key = (row["country"], match_bucket(len(truth[entity_id])),
               ",".join(sorted(scripts(row["business_name"]))),
               "common" if core and names[row["country"], core] >= 3 else "other")
        groups[key].append(entity_id)
    # Proportional allocation with a minimum of one when the pilot permits it.
    quota = {key: min(len(ids), max(1, int(size * len(ids) / len(source1))))
             for key, ids in groups.items()}
    while sum(quota.values()) > size:
        key = max(quota, key=lambda k: (quota[k] - size * len(groups[k]) / len(source1), k))
        quota[key] -= 1
    while sum(quota.values()) < size:
        key = max((k for k in quota if quota[k] < len(groups[k])),
                  key=lambda k: (size * len(groups[k]) / len(source1) - quota[k], k))
        quota[key] += 1
    rng = random.Random(seed)
    selected, strata = [], []
    for key, ids in sorted(groups.items()):
        ordered = sorted(ids)
        rng.shuffle(ordered)
        selected.extend(ordered[:quota[key]])
        strata.append({"stratum": key, "available": len(ids), "selected": quota[key]})
    return sorted(selected), strata


def digest_files(paths):
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in paths]


class ProjectedIndex:
    def __init__(self, query_keys, key_function=legacy_record_keys, cap_function=_cap):
        self.key_function = key_function
        self.df = dict.fromkeys(query_keys, 0)
        self.tf = dict.fromkeys(query_keys, 0)
        self.caps = {key: cap_function(key.split("|", 1)[1]) for key in query_keys}
        self.postings = {}
        self.country_n = Counter()

    def add(self, row):
        country = row["country"].strip()
        self.country_n[country] += 1
        frequencies = Counter(k for k, _ in self.key_function(row["business_name"], row["business_address"]))
        self.add_frequencies(row["entity_id"], {country + "|" + key: count
                                              for key, count in frequencies.items()
                                              if country + "|" + key in self.df})

    def add_frequencies(self, entity_id, frequencies):
        packed = None
        for full, multiplicity in frequencies.items():
            self.df[full] += 1
            self.tf[full] += multiplicity
            if self.df[full] > self.caps[full]:
                self.postings.pop(full, None)
            else:
                if packed is None:
                    packed = pack_id(entity_id)
                self.postings.setdefault(full, array("Q")).extend([packed] * multiplicity)

    def usable(self, key, legacy=False):
        count = self.tf[key] if legacy else self.df[key]
        return 0 < count <= self.caps[key] and key in self.postings

    def rank(self, row, legacy=False, with_routes=True):
        country = row["country"].strip()
        keys = self.key_function(row["business_name"], row["business_address"])
        if not legacy:
            keys = unique_keys(keys)
        scores = defaultdict(float)
        routes = defaultdict(set)
        for key, weight in keys:
            full = country + "|" + key
            if not self.usable(full, legacy):
                continue
            df = self.tf[full] if legacy else self.df[full]
            contribution = weight * math.log((self.country_n[country] + 1) / (df + 1))
            bucket = self.postings[full]
            for candidate in (bucket if legacy else dict.fromkeys(bucket)):
                scores[candidate] += contribution
                if with_routes:
                    parts = key.split(":", 2)
                    routes[candidate].add("f_" + parts[1] if parts[0] == "f" else parts[0])
        ranked = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
        return ranked, scores, routes

    def save(self, path, metadata, targets):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite cache: {path}")
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE metadata (value TEXT)")
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps({"signature": metadata, "country_n": self.country_n, "targets": targets}),))
            db.execute("CREATE TABLE keys (key TEXT PRIMARY KEY, df INTEGER, tf INTEGER, cap INTEGER, posting BLOB)")
            db.executemany("INSERT INTO keys VALUES (?,?,?,?,?)", (
                (key, self.df[key], self.tf[key], self.caps[key], self.postings.get(key, array("Q")).tobytes())
                for key in self.df))

    @classmethod
    def load(cls, path, metadata, key_function=legacy_record_keys):
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
            saved = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            if saved["signature"] != metadata:
                raise ValueError("Cache signature differs; use a new --cache path")
            index = cls([], key_function=key_function)
            index.country_n.update(saved["country_n"])
            for key, df, tf, cap, blob in db.execute("SELECT * FROM keys"):
                index.df[key], index.tf[key], index.caps[key] = df, tf, cap
                if blob:
                    index.postings[key] = array("Q")
                    index.postings[key].frombytes(blob)
        return index, saved["targets"]


def _worker_init(query_keys, truth_ids, key_function):
    global _worker_keys, _worker_truth, _worker_function, _worker_batches
    _worker_keys, _worker_truth, _worker_function = set(query_keys), set(truth_ids), key_function
    _worker_batches = 0


def _worker_batch(batch):
    global _worker_batches
    countries, records, targets = Counter(), [], {}
    for row in batch:
        country = row["country"].strip()
        countries[country] += 1
        frequencies = Counter(country + "|" + key for key, _ in
                              _worker_function(row["business_name"], row["business_address"])
                              if country + "|" + key in _worker_keys)
        if frequencies:
            records.append((row["entity_id"], frequencies))
        if row["entity_id"] in _worker_truth:
            targets[row["entity_id"]] = row
    _worker_batches += 1
    memory = (os.getpid(), rss_mb()) if _worker_batches % 100 == 1 else None
    return countries, records, targets, memory


def scan_projected(index, paths, true_ids, workers=1, batch_size=2000):
    """Bounded ordered work queue; workers emit only projected key frequencies."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    targets = {}
    index.scan_profile = {"worker_count": workers}
    if workers == 1:
        for path in paths:
            for i, row in enumerate(rows(path), 1):
                index.add(row)
                if row["entity_id"] in true_ids:
                    targets[row["entity_id"]] = row
                if i % 250000 == 0:
                    print(f"scan {path.name}: {i:,}", flush=True)
        return targets

    def batches():
        for path in paths:
            batch = []
            for row in rows(path):
                batch.append(row)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    iterator, pending = iter(batches()), deque()
    processed, next_log = 0, 250000
    worker_peaks = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(list(index.df), true_ids, index.key_function)) as executor:
        def submit_next():
            batch = next(iterator, None)
            if batch is not None:
                pending.append(executor.submit(_worker_batch, batch))
        for _ in range(workers * 2):
            submit_next()
        while pending:
            countries, records, found, memory = pending.popleft().result()
            if memory:
                worker_peaks[memory[0]] = memory[1]["process_peak_rss_mb"]
            index.country_n.update(countries)
            targets.update(found)
            for entity_id, frequencies in records:
                index.add_frequencies(entity_id, frequencies)
            processed += sum(countries.values())
            if processed >= next_log:
                print(f"parallel scan: {processed:,} rows ({workers} workers)", flush=True)
                next_log += 250000
            submit_next()
    index.scan_profile["sum_worker_sampled_peak_rss_mb"] = sum(worker_peaks.values())
    index.scan_profile["memory_note"] = "Phase RSS is parent only; worker peak sum is separately sampled, not simultaneous process-tree peak."
    return targets


def distribution(values):
    values = sorted(values)
    return {"count": len(values), **{name: values[min(len(values)-1, int((len(values)-1)*q))] if values else 0
            for name, q in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1))}}


def classify(raw, usable, rank, k):
    if not raw:
        return "NO_RAW_KEY"
    if not usable:
        return "ALL_SHARED_KEYS_PRUNED"
    if rank is None:
        raise AssertionError("Usable shared key must put truth in ranked pool")
    return "IN_TOP_K" if rank <= k else "RANKED_BELOW_K"


def measure(index, source1, truth, targets, output, legacy=False, reason_k=100):
    groups = defaultdict(lambda: {"entities": 0, "true_links": 0, "singletons": 0,
                                  "zero_usable_keys": 0, "reasons": Counter(),
                                  "at_k": {str(k): {"hits": 0, "oracle_sum": 0., "covered": 0} for k in (*KS, "all")}})
    sizes = []
    family_sizes = defaultdict(list)
    key_fn = index.key_function
    fields = ["source1_id", "true_candidate_id", "country", "candidate_source", "shared_raw_families",
              "shared_usable_families", "reason", "rank_if_ranked", "retrieval_score", "number_of_true_siblings",
              "script_mismatch", "missing_address"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, (entity_id, row) in enumerate(sorted(source1.items()), 1):
            country = row["country"].strip()
            ranked, scores, _routes = index.rank(row, legacy, with_routes=False)
            rank_map = {candidate: rank for rank, candidate in enumerate(ranked, 1)}
            labels = truth[entity_id]
            sizes.append(len(ranked))
            keys = {key for key, _ in key_fn(row["business_name"], row["business_address"])}
            usable = {key for key in keys if index.usable(country + "|" + key, legacy)}
            for key in keys:
                full = country + "|" + key
                family_sizes[key.split(":", 1)[0]].append(index.tf[full] if legacy else index.df[full])
            for group in ("all", "country:" + country):
                g = groups[group]
                g["entities"] += 1
                g["true_links"] += len(labels)
                g["singletons"] += not labels
                g["zero_usable_keys"] += not usable
                for k in (*KS, "all"):
                    hits = {label for label in labels if pack_id(label) in rank_map and
                            (k == "all" or rank_map[pack_id(label)] <= k)}
                    bucket = g["at_k"][str(k)]
                    bucket["hits"] += len(hits)
                    bucket["oracle_sum"] += entity_f05(labels, hits)
                    bucket["covered"] += bool(hits)
            for label in sorted(labels):
                target = targets[label]
                other_keys = {key for key, _ in key_fn(target["business_name"], target["business_address"])}
                raw = keys & other_keys
                shared = raw & usable if target["country"].strip() == country else set()
                rank = rank_map.get(pack_id(label))
                reason = classify(raw, shared, rank, reason_k)
                for group in ("source:" + label[:2], "country_source:" + country + ":" + label[:2]):
                    groups[group]["true_links"] += 1
                    for k in (*KS, "all"):
                        groups[group]["at_k"][str(k)]["hits"] += rank is not None and (k == "all" or rank <= k)
                for group in ("all", "country:" + country, "source:" + label[:2], "country_source:" + country + ":" + label[:2]):
                    groups[group]["reasons"][reason] += 1
                writer.writerow(dict(zip(fields, [entity_id, label, country, label[:2],
                    ",".join(sorted({k.split(":", 1)[0] for k in raw})),
                    ",".join(sorted({k.split(":", 1)[0] for k in shared})), reason, rank or "",
                    scores.get(pack_id(label), ""), len(labels),
                    int(scripts(row["business_name"]) != scripts(target["business_name"])),
                    int(not row["business_address"].strip() or not target["business_address"].strip())])))
            if i % 250 == 0:
                print(f"{'legacy' if legacy else 'dedup'}: queried {i:,}", flush=True)
    for name, group in groups.items():
        for bucket in group["at_k"].values():
            bucket["pair_recall"] = bucket["hits"] / group["true_links"] if group["true_links"] else None
            oracle_sum = bucket.pop("oracle_sum")
            bucket["oracle_macro_f05"] = oracle_sum / group["entities"] if group["entities"] else None
        assert sum(group["reasons"].values()) == group["true_links"]
    return {"groups": dict(groups), "pool_size": distribution(sizes),
            "posting_sizes_per_query_key_by_family": {k: distribution(v) for k, v in family_sizes.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--restore-split", action="store_true", help="Recreate missing frozen split using original algorithm")
    parser.add_argument("--output", type=Path, default=config.REPORTS_DIR / "retrieval_pilot")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Write fixed pilot membership without building a baseline index")
    args = parser.parse_args()
    timer = Timer()
    if not SPLIT_PATH.exists():
        if not args.restore_split:
            parser.error("Frozen split missing; pass --restore-split to reproduce it with seed 2026")
        print(build_split(), flush=True)
    timer.mark("split")
    dev_ids = {r["source1_entity_id"] for r in rows(SPLIT_PATH) if r["dev_sample"] == "1" and r["split"] == "valid"}
    source1 = load_selected(config.TRAIN_SOURCE1_PATH, dev_ids)
    raw_truth = load_selected(config.TRAIN_GROUND_TRUTH_PATH, dev_ids, "source1_entity_id")
    truth = {key: set(row["matched_entity_ids"].split(",")) - {""} for key, row in raw_truth.items()}
    selected, strata = select_pilot(source1, truth, args.size, args.seed)
    source1 = {key: source1[key] for key in selected}
    truth = {key: truth[key] for key in selected}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, delimiter="\t", lineterminator="\n")
    writer.writerow(["source1_entity_id", "country", "match_count"])
    writer.writerows((key, source1[key]["country"], len(truth[key])) for key in selected)
    manifest_path = args.output / "pilot_ids.tsv"
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != manifest.getvalue():
        manifest_path.write_text(manifest.getvalue(), encoding="utf-8", newline="")
    if args.prepare_only:
        preparation = {"seed": args.seed, "pilot_size": len(selected), "strata": strata,
                       "true_links": sum(map(len, truth.values())),
                       "manifest_sha256": hashlib.sha256(manifest.getvalue().encode()).hexdigest()}
        (args.output / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {len(selected):,} entities in {manifest_path}", flush=True)
        return
    query_keys = {row["country"].strip() + "|" + key for row in source1.values()
                  for key, _ in record_keys(row["business_name"], row["business_address"])}
    signature = {"version": VERSION, "byteorder": sys.byteorder,
                 "files": digest_files([config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH,
                                         config.TRAIN_SOURCE3_PATH, config.TRAIN_GROUND_TRUTH_PATH, SPLIT_PATH]),
                 "keys_hash": hashlib.sha256("\n".join(sorted(query_keys)).encode()).hexdigest(),
                 "pilot_hash": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
                 "code_hash": hashlib.sha256(b"".join((Path(__file__).parent / f).read_bytes()
                     for f in ("retrieve_v2.py", "normalize.py", "block.py"))).hexdigest()}
    timer.mark("select_pilot")
    cache = args.cache or args.output / "projected_index.sqlite3"
    if cache.exists():
        index, targets = ProjectedIndex.load(cache, signature)
        timer.mark("load_cache")
    else:
        index = ProjectedIndex(query_keys)
        targets = {}
        true_ids = set().union(*truth.values())
        for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
            for i, row in enumerate(rows(path), 1):
                index.add(row)
                if row["entity_id"] in true_ids:
                    targets[row["entity_id"]] = row
                if i % 250_000 == 0:
                    print(f"scan {path.name}: {i:,} {rss_mb()}", flush=True)
        if set(targets) != true_ids:
            raise ValueError("Some true S2/S3 IDs are missing from training inputs")
        timer.mark("count_and_bounded_postings")
        cache.parent.mkdir(parents=True, exist_ok=True)
        index.save(cache, signature, targets)
        timer.mark("save_cache")
    report = {"signature": signature, "seed": args.seed, "pilot_size": len(selected),
              "strata": strata, "reason_k": 100,
              "note": "All true siblings; proportional country/count/script/common-name pilot from frozen dev. Oracle includes singletons. Counts use full training S2/S3; no test data."}
    for legacy in (True, False):
        mode = "legacy" if legacy else "dedup"
        report[mode] = measure(index, source1, truth, targets, args.output / f"losses_{mode}.tsv", legacy)
        timer.mark("query_" + mode)
        print(mode, json.dumps(report[mode]["groups"]["all"], indent=2), flush=True)
    report["phases"] = timer.phases
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
