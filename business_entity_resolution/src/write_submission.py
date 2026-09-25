"""Write test matching_results.tsv with the best measured rule.

Rule: token-and-number candidates, plus name tokens that appear on at most 20
records, kept when the pair score is at least 0.75. On the frozen dev sample
that cutoff scored macro F0.5 0.4389. This is a baseline upload, not the
retrieval-v2 oracle.
"""

import csv
import sys

from . import config
from .block import blocking_keys
from .rank import _candidates, _pair_score, _signature

THRESHOLD = 0.75
MAX_TOKEN = 15_000
_TOO_COMMON = object()


def _index_test() -> tuple[dict[str, list[str]], dict[str, tuple[tuple[str, ...], tuple[str, ...]]]]:
    postings: dict[str, list[str] | object] = {}
    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    def add(key: str, entity_id: str) -> None:
        slot = postings.get(key)
        if slot is _TOO_COMMON:
            return
        if slot is None:
            postings[key] = [entity_id]
            return
        slot.append(entity_id)  # type: ignore[union-attr]
        if len(slot) > MAX_TOKEN:  # type: ignore[arg-type]
            postings[key] = _TOO_COMMON

    for path in (config.TEST_SOURCE2_PATH, config.TEST_SOURCE3_PATH):
        print(f"Indexing {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
                country = row["country"].strip()
                entity_id = row["entity_id"]
                signatures[entity_id] = _signature(row["business_name"], row["business_address"])
                for key in blocking_keys(row["business_name"], row["business_address"]):
                    if key.startswith("t:") or (key.startswith("n:") and len(key) >= 5):
                        add(country + "|" + key, entity_id)
                if i % 1_000_000 == 0:
                    print(f"  {i:,} rows", flush=True)
    usable = {key: value for key, value in postings.items() if value is not _TOO_COMMON}
    print(f"Usable keys: {len(usable):,}  signatures: {len(signatures):,}", flush=True)
    return usable, signatures  # type: ignore[return-value]


def main() -> int:
    postings, signatures = _index_test()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = config.OUTPUT_DIR / "matching_results.tsv"
    rows = 0
    nonempty = 0
    with config.TEST_SOURCE1_PATH.open("r", encoding="utf-8", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for row in reader:
            entity_id = row["entity_id"]
            country = row["country"].strip()
            signature = _signature(row["business_name"], row["business_address"])
            keys = blocking_keys(row["business_name"], row["business_address"])
            chosen: list[tuple[float, str]] = []
            for candidate_id in _candidates(country, keys, postings):
                other = signatures.get(candidate_id)
                if other is None:
                    continue
                score = _pair_score(signature, other)
                if score >= THRESHOLD:
                    chosen.append((score, candidate_id))
            chosen.sort(key=lambda item: (-item[0], item[1]))
            seen: set[str] = set()
            matched: list[str] = []
            for _score, candidate_id in chosen:
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                matched.append(candidate_id)
            writer.writerow([entity_id, ",".join(matched)])
            rows += 1
            nonempty += bool(matched)
            if rows % 200_000 == 0:
                print(f"  wrote {rows:,}  with matches {nonempty:,}", flush=True)
    print(f"Rows: {rows:,}  with at least one match: {nonempty:,}", flush=True)
    print(f"Saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
