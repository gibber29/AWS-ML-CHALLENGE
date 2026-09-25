"""Rank a short blocked list with name overlap and shared address numbers.

Candidate generation walks the rarest name keys first and keeps at most 80
records. The score is token Jaccard plus a bonus when address numbers agree.
Several thresholds are scored on the frozen dev sample in one pass.
"""

from collections import Counter
import csv
import sys

from . import config
from .block import _core_tokens, _dev_ids, _entity_f05, _load_truth, _numbers, blocking_keys

SHORTLIST = 80
MAX_TOKEN = 15_000
MAX_PREFIX = 1_500
_TOO_COMMON = object()


def _signature(name: str, address: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tokens = tuple(dict.fromkeys(_core_tokens(name)))
    numbers = tuple(dict.fromkeys(number for number in _numbers(address) if len(number) >= 2))
    return tokens, numbers


def _pair_score(left: tuple[tuple[str, ...], tuple[str, ...]], right: tuple[tuple[str, ...], tuple[str, ...]]) -> float:
    left_tokens, left_numbers = set(left[0]), set(left[1])
    right_tokens, right_numbers = set(right[0]), set(right[1])
    union = left_tokens | right_tokens
    name = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
    shared = left_numbers & right_numbers
    bonus = 0.0
    if any(len(number) >= 4 for number in shared):
        bonus = 0.45
    elif any(len(number) >= 3 for number in shared):
        bonus = 0.30
    elif shared:
        bonus = 0.12
    return name + bonus


def _index() -> tuple[dict[str, list[str]], dict[str, tuple[tuple[str, ...], tuple[str, ...]]]]:
    postings: dict[str, list[str] | object] = {}
    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    def add(key: str, entity_id: str, limit: int) -> None:
        slot = postings.get(key)
        if slot is _TOO_COMMON:
            return
        if slot is None:
            postings[key] = [entity_id]
            return
        slot.append(entity_id)  # type: ignore[union-attr]
        if len(slot) > limit:  # type: ignore[arg-type]
            postings[key] = _TOO_COMMON

    for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        print(f"Indexing {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
                country = row["country"].strip()
                entity_id = row["entity_id"]
                signature = _signature(row["business_name"], row["business_address"])
                signatures[entity_id] = signature
                for key in blocking_keys(row["business_name"], row["business_address"]):
                    if key.startswith("t:"):
                        add(country + "|" + key, entity_id, MAX_TOKEN)
                    elif key.startswith("p4:"):
                        add(country + "|" + key, entity_id, MAX_PREFIX)
                    elif key.startswith("n:") and len(key) >= 5:
                        add(country + "|" + key, entity_id, MAX_TOKEN)
                if i % 1_000_000 == 0:
                    print(f"  {i:,} rows", flush=True)
    usable = {key: value for key, value in postings.items() if value is not _TOO_COMMON}
    print(f"Usable keys: {len(usable):,}  signatures: {len(signatures):,}", flush=True)
    return usable, signatures  # type: ignore[return-value]


def _load_source1(dev_ids: set[str]) -> dict[str, tuple[str, tuple[tuple[str, ...], tuple[str, ...]], set[str]]]:
    records = {}
    with config.TRAIN_SOURCE1_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entity_id = row["entity_id"]
            if entity_id not in dev_ids:
                continue
            keys = blocking_keys(row["business_name"], row["business_address"])
            records[entity_id] = (
                row["country"].strip(),
                _signature(row["business_name"], row["business_address"]),
                keys,
            )
            if len(records) == len(dev_ids):
                break
    return records


def _candidates(country: str, keys: set[str], postings: dict[str, list[str]]) -> list[str]:
    """Records that share both a name token and an address number."""
    token_sets: list[set[str]] = []
    number_sets: list[set[str]] = []
    chosen: list[str] = []
    seen: set[str] = set()
    for key in keys:
        bucket = postings.get(country + "|" + key)
        if not bucket:
            continue
        if key.startswith("t:") and len(key) >= 7:
            token_sets.append(set(bucket))
            if len(bucket) <= 20:
                for entity_id in bucket:
                    if entity_id not in seen:
                        seen.add(entity_id)
                        chosen.append(entity_id)
        elif key.startswith("n:") and len(key) >= 5:
            number_sets.append(set(bucket))
    for token_ids in token_sets:
        for number_ids in number_sets:
            hit = token_ids & number_ids
            if not hit or len(hit) > 30:
                continue
            for entity_id in hit:
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                chosen.append(entity_id)
    return chosen


def main() -> int:
    dev_ids = _dev_ids()
    print(f"Dev entities: {len(dev_ids):,}", flush=True)
    source1 = _load_source1(dev_ids)
    truth = _load_truth(dev_ids)
    postings, signatures = _index()

    thresholds = (0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
    totals = {threshold: 0.0 for threshold in thresholds}
    totals["top1>=0.60"] = 0.0
    totals["top3>=0.45"] = 0.0
    retrieved = 0
    true_links = 0
    sizes: list[int] = []
    for i, (entity_id, (country, signature, keys)) in enumerate(source1.items(), start=1):
        labels = truth.get(entity_id, set())
        true_links += len(labels)
        scored: list[tuple[float, str]] = []
        for candidate_id in _candidates(country, keys, postings):
            other = signatures.get(candidate_id)
            if other is None:
                continue
            scored.append((_pair_score(signature, other), candidate_id))
        scored.sort(reverse=True)
        sizes.append(len(scored))
        retrieved += sum(1 for _score, candidate_id in scored if candidate_id in labels)
        for threshold in thresholds:
            guess = {candidate_id for score, candidate_id in scored if score >= threshold}
            totals[threshold] += _entity_f05(labels, guess)
        totals["top1>=0.60"] += _entity_f05(labels, {scored[0][1]} if scored and scored[0][0] >= 0.60 else set())
        totals["top3>=0.45"] += _entity_f05(
            labels, {candidate_id for score, candidate_id in scored[:3] if score >= 0.45}
        )
        if i % 10_000 == 0:
            print(f"  scored {i:,}", flush=True)

    count = len(source1) or 1
    sizes.sort()
    lines = [
        "PAIR-SCORE SWEEP",
        f"Source 1 scored: {len(source1):,}",
        f"True links: {true_links:,}",
        f"Shortlist pair-recall: {retrieved / true_links if true_links else 0:.4f} ({retrieved:,})",
        f"Median shortlist: {sizes[len(sizes) // 2] if sizes else 0:,}",
    ]
    best_name = ""
    best_value = -1.0
    for name, total in totals.items():
        macro = total / count
        lines.append(f"{name}: macro F0.5 {macro:.4f}")
        if macro > best_value:
            best_name = str(name)
            best_value = macro
    lines.append(f"Best: {best_name} {best_value:.4f}")
    text = "\n".join(lines)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.REPORTS_DIR / "rank_sweep.txt"
    output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"Saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
