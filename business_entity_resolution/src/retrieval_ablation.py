"""Reuse an enhanced pilot index to measure each additional route independently."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from . import config
from .retrieval_pilot import ProjectedIndex, Timer, rows, load_selected, digest_files, measure
from .retrieve_v3 import enhanced_keys, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=config.REPORTS_DIR / "retrieval_pilot")
    parser.add_argument("--index-dir", type=Path, default=config.REPORTS_DIR / "retrieval_v3_pilot")
    args = parser.parse_args()
    timer = Timer()
    ids = {r["source1_entity_id"] for r in rows(args.pilot_dir / "pilot_ids.tsv")}
    source1 = load_selected(config.TRAIN_SOURCE1_PATH, ids)
    raw_truth = load_selected(config.TRAIN_GROUND_TRUTH_PATH, ids, "source1_entity_id")
    truth = {key: set(row["matched_entity_ids"].split(",")) - {""} for key, row in raw_truth.items()}
    path = args.index_dir / "projected_index.sqlite3"
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        signature = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])["signature"]
    if signature["version"] != VERSION:
        raise ValueError("Index key version changed")
    code_hash = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                for name in ("retrieve_v3.py", "retrieve_v2.py", "normalize.py", "block.py"))).hexdigest()
    if code_hash != signature["code_hash"]:
        raise ValueError("Index normalization/key code changed")
    for stamp in signature["files"]:
        if digest_files([Path(stamp["path"])])[0] != stamp:
            raise ValueError("Index input file changed")
    keys = {r["country"].strip() + "|" + key for r in source1.values()
            for key, _ in enhanced_keys(r["business_name"], r["business_address"])}
    if hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest() != signature["key_hash"]:
        raise ValueError("Pilot membership/keys changed")
    index, targets = ProjectedIndex.load(path, signature, enhanced_keys)
    timer.mark("load")
    report = {"signature": signature, "note": "Same pilot and full-corpus counts; each route added independently to corrected baseline."}
    base = {"t", "p", "s", "sk", "c", "n", "ns", "aw"}
    for mode, families in (("baseline", base), ("folded", base | {"f"}),
                           ("address_combinations", base | {"a2", "an", "n2"}),
                           ("skeleton_fragments", base | {"sg"})):
        def key_function(name, address):
            return [(key, weight) for key, weight in enhanced_keys(name, address)
                    if key.split(":", 1)[0] in families]
        index.key_function = key_function
        report[mode] = measure(index, source1, truth, targets, args.index_dir / f"ablation_{mode}_losses.tsv")
        timer.mark(mode)
        print(mode, json.dumps(report[mode]["groups"]["all"], indent=2), flush=True)
    report["phases"] = timer.phases
    (args.index_dir / "ablations.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
