"""High-recall retrieval. Several independent routes, then a ranked shortlist.

The old ranker kept a candidate only when a name token and an address number
agreed, which held pair-recall at 0.37. This module unions exact tokens,
prefixes, suffixes, a consonant skeleton, a compact name, address numbers,
number sequences, and address words. Keys seen too often are dropped.
Candidates are ranked by inverse document frequency and cut to K.

The oracle score predicts exactly the true links that landed in that shortlist.
It is the ceiling of a perfect classifier on this list, not a leaderboard score.
"""

from collections import Counter, defaultdict
import csv
import math
import sys
import time
import tracemalloc

from . import config
from .block import _core_tokens, _dev_ids, _load_truth, _numbers
from .evaluate import entity_f05
from .normalize import consonant_skeleton

KS = (10, 20, 50, 100)
_PUNCT = str.maketrans({chr(i): " " for i in range(128) if not chr(i).isalnum() and chr(i) != " "})
_ADDR_STOP = {
    "street", "stree", "road", "lane", "drive", "avenue", "court", "place",
    "floor", "near", "plot", "nagar", "road", "main", "cross", "block",
    "phase", "sector", "house", "building", "city", "town", "west", "east",
    "north", "south", "india", "state",
}
# key family -> max documents kept in one country
_CAPS = {"t": 4000, "p": 1800, "s": 1800, "sk": 5000, "c": 600, "n": 5000, "ns": 2500, "aw": 1200}


def _latin_skeleton(tokens: list[str]) -> str:
    chars: list[str] = []
    for token in tokens:
        for char in token:
            if char in "aeiouh":
                continue
            chars.append("ks" if char == "x" else "k" if char == "c" else char)
    return "".join(chars)


def legacy_record_keys(name: str, address: str) -> list[tuple[str, float]]:
    """Country-independent keys and their type weights."""
    name = "" if name is None else str(name)
    address = "" if address is None else str(address)
    tokens = _core_tokens(name)
    if not tokens:
        rough = name.casefold().translate(_PUNCT).split()
        tokens = [token for token in rough if len(token) >= 2 and token.isascii()]
    keys: list[tuple[str, float]] = []
    for token in tokens:
        keys.append((f"t:{token}", 3.0))
        if len(token) >= 5:
            keys.append((f"p:{token[:4]}", 1.0))
            keys.append((f"s:{token[-4:]}", 1.4))
    if any(ord(char) > 127 for char in name):
        skeleton = consonant_skeleton(name.casefold())
    else:
        skeleton = _latin_skeleton(tokens)
    if len(skeleton) >= 8:
        keys.append((f"sk:{skeleton[:8]}", 2.6))
    elif len(skeleton) >= 5:
        keys.append((f"sk:{skeleton}", 2.6))
    compact = "".join(tokens)
    if len(compact) >= 6:
        keys.append((f"c:{compact[:12]}", 2.8))
    numbers = []
    for number in _numbers(address):
        if len(number) >= 2:
            numbers.append(number)
            weight = 2.3 if len(number) >= 4 else 1.1 if len(number) >= 3 else 0.6
            keys.append((f"n:{number}", weight))
    if len(numbers) >= 2:
        keys.append((f"ns:{'-'.join(numbers[:4])}", 2.5))
    for word in address.casefold().translate(_PUNCT).split():
        if len(word) >= 5 and word.isascii() and not word.isdigit() and word not in _ADDR_STOP:
            keys.append((f"aw:{word}", 0.8))
    return keys


def unique_keys(keys: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """One contribution per document/key; the strongest explicit weight wins."""
    weights: dict[str, float] = {}
    for key, weight in keys:
        weights[key] = max(weight, weights.get(key, weight))
    return list(weights.items())


def record_keys(name: str, address: str) -> list[tuple[str, float]]:
    return unique_keys(legacy_record_keys(name, address))


def _family(key: str) -> str:
    return key.split(":", 1)[0]


def _cap(key: str) -> int:
    if key.startswith("n:") and len(key) <= 4:
        return 400
    return _CAPS.get(_family(key), 2000)


def pack_id(entity_id: str) -> int:
    kind = 0 if entity_id.startswith("S2-") else 1
    return (kind << 40) | int(entity_id.split("-", 1)[1])


def unpack_id(packed: int) -> str:
    kind = packed >> 40
    number = packed & ((1 << 40) - 1)
    return f"S{2 + kind}-{number}"


def index_rows(rows: list[tuple[str, str, str, str]]) -> tuple[dict[str, list[int]], dict[str, int], Counter]:
    """Index an in-memory row list. Each row is id, name, address, country."""
    counts: Counter[str] = Counter()
    parsed: list[tuple[str, str, list[tuple[str, float]]]] = []
    for entity_id, name, address, country in rows:
        if not entity_id.startswith(("S2-", "S3-")):
            continue
        keys = record_keys(name, address)
        parsed.append((entity_id, country, keys))
        for key, _weight in keys:
            counts[f"{country}|{key}"] += 1
    postings: dict[str, list[int]] = {}
    for entity_id, country, keys in parsed:
        if not (entity_id.startswith("S2-") or entity_id.startswith("S3-")):
            continue
        packed = pack_id(entity_id)
        for key, _weight in keys:
            full = f"{country}|{key}"
            if counts[full] <= _cap(key):
                postings.setdefault(full, []).append(packed)
    country_n: Counter = Counter(country for _id, _name, _address, country in rows if _id.startswith(("S2-", "S3-")))
    return postings, counts, country_n


def retrieve(
    country: str,
    keys: list[tuple[str, float]],
    postings: dict[str, list[int]],
    counts: dict[str, int],
    country_n: int,
    k: int,
) -> list[str]:
    scores: dict[int, float] = {}
    for key, weight in unique_keys(keys):
        full = f"{country}|{key}"
        bucket = postings.get(full)
        if not bucket:
            continue
        df = counts.get(full, len(bucket))
        idf = weight * math.log((country_n + 1) / (df + 1))
        for packed in bucket:
            scores[packed] = scores.get(packed, 0.0) + idf
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [unpack_id(packed) for packed, _score in ranked[:k]]


def _count_files() -> tuple[Counter, Counter]:
    counts: Counter[str] = Counter()
    country_n: Counter = Counter()
    for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        print(f"Counting {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
                country = row["country"].strip()
                country_n[country] += 1
                for key, _weight in record_keys(row["business_name"], row["business_address"]):
                    counts[f"{country}|{key}"] += 1
                if i % 1_000_000 == 0:
                    print(f"  {i:,} rows, {len(counts):,} keys", flush=True)
    return counts, country_n


def _build_postings(counts: Counter) -> dict[str, list[int]]:
    kept = {key for key, df in counts.items() if 0 < df <= _cap(key.split("|", 1)[1])}
    print(f"Keys kept: {len(kept):,} of {len(counts):,}", flush=True)
    postings: dict[str, list[int]] = defaultdict(list)
    for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        print(f"Indexing {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
                country = row["country"].strip()
                packed = pack_id(row["entity_id"])
                for key, _weight in record_keys(row["business_name"], row["business_address"]):
                    full = f"{country}|{key}"
                    if full in kept:
                        postings[full].append(packed)
                if i % 1_000_000 == 0:
                    print(f"  {i:,} rows", flush=True)
    return postings


def _load_dev(dev_ids: set[str]) -> dict[str, tuple[str, list[tuple[str, float]]]]:
    records = {}
    with config.TRAIN_SOURCE1_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entity_id = row["entity_id"]
            if entity_id not in dev_ids:
                continue
            records[entity_id] = (
                row["country"].strip(),
                record_keys(row["business_name"], row["business_address"]),
            )
            if len(records) == len(dev_ids):
                break
    return records


def evaluate_dev() -> str:
    tracemalloc.start()
    started = time.perf_counter()
    dev_ids = _dev_ids()
    print(f"Dev entities: {len(dev_ids):,}", flush=True)
    source1 = _load_dev(dev_ids)
    truth = _load_truth(dev_ids)
    counts, country_n = _count_files()
    postings = _build_postings(counts)
    hits = {k: 0 for k in KS}
    oracle = {k: 0.0 for k in KS}
    covered = {k: 0 for k in KS}
    by_country = {name: {k: [0, 0] for k in KS} for name in ("US", "India")}
    source_hits = {"S2": 0, "S3": 0}
    source_total = {"S2": 0, "S3": 0}
    true_links = 0
    sizes: list[int] = []
    zero = 0
    for i, (entity_id, (country, keys)) in enumerate(source1.items(), start=1):
        labels = truth.get(entity_id, set())
        true_links += len(labels)
        for label in labels:
            if label.startswith("S2-"):
                source_total["S2"] += 1
            elif label.startswith("S3-"):
                source_total["S3"] += 1
        n_country = country_n[country]
        ranked = retrieve(country, keys, postings, counts, n_country, max(KS))
        sizes.append(len(ranked))
        if not ranked:
            zero += 1
        found = set(ranked)
        for label in labels:
            if label in found:
                if label.startswith("S2-"):
                    source_hits["S2"] += 1
                elif label.startswith("S3-"):
                    source_hits["S3"] += 1
        for k in KS:
            top = set(ranked[:k])
            got = len(labels & top)
            hits[k] += got
            if got:
                covered[k] += 1
            oracle[k] += entity_f05(labels, labels & top)
            if country in by_country:
                by_country[country][k][0] += got
                by_country[country][k][1] += len(labels)
        if i % 5_000 == 0:
            print(f"  queried {i:,}", flush=True)
    _current, peak = tracemalloc.get_traced_memory()
    elapsed = time.perf_counter() - started
    sizes.sort()
    def pct(index: int) -> int:
        if not sizes:
            return 0
        return sizes[min(len(sizes) - 1, index)]

    lines = [
        "RETRIEVE V2 — frozen 50,000 dev entities",
        f"Seconds: {elapsed:.1f}",
        f"Peak traced memory MB: {peak / (1024 * 1024):.0f}",
        f"Posting keys: {len(postings):,}",
        f"True links: {true_links:,}",
        f"Entities with zero candidates at K={max(KS)}: {zero:,} / {len(source1):,}",
        f"Shortlist size p50/p90/p99/max at K={max(KS)}: "
        f"{pct(len(sizes)//2):,} / {pct(int(len(sizes)*0.9)):,} / {pct(int(len(sizes)*0.99)):,} / {sizes[-1] if sizes else 0:,}",
        f"S2 recall@{max(KS)}: {source_hits['S2']:,} / {source_total['S2']:,}",
        f"S3 recall@{max(KS)}: {source_hits['S3']:,} / {source_total['S3']:,}",
    ]
    for k in KS:
        recall = hits[k] / true_links if true_links else 0.0
        ceiling = oracle[k] / len(source1) if source1 else 0.0
        lines.append(
            f"K={k}: pair-recall {recall:.4f} ({hits[k]:,}/{true_links:,})  "
            f"entities with >=1 true hit {covered[k]:,}/{len(source1):,}  "
            f"oracle macro F0.5 {ceiling:.4f}"
        )
        for name, buckets in by_country.items():
            got, total = buckets[k]
            lines.append(f"  {name} pair-recall@{k}: {got:,} / {total:,} = {got / total if total else 0:.4f}")
    text = "\n".join(lines)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.REPORTS_DIR / "retrieve_v2_dev50k.txt"
    output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"Saved {output}", flush=True)
    return text


def main() -> int:
    evaluate_dev()
    return 0


if __name__ == "__main__":
    sys.exit(main())
