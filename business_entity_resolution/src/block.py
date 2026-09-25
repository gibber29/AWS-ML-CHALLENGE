"""Same-country blocking keys and a dev-sample recall measurement.

Keys come from the hand-read pairs: a shared name token or its 4-character
prefix, a consonant skeleton (so Indic and Latin spellings meet), a
space-stripped name prefix (so "urban nails" meets "urbannails.com"), and
address numbers with leading zeros removed.
"""

from collections import Counter
import csv
from dataclasses import dataclass
import string

from . import config
from .normalize import consonant_skeleton
from .split import SPLIT_PATH

_PUNCT = str.maketrans({char: " " for char in string.punctuation})
_SUFFIXES = {
    "inc", "ltd", "llc", "llp", "plc", "corp", "co", "company", "limited",
    "corporation", "incorporated", "pvt", "private", "sarl", "sas", "sa",
    "gmbh", "pllc", "ltda",
}

MAX_POSTING = 15_000
MAX_TOKEN_POSTING = 15_000
SHORTLIST = 40
_TOO_COMMON = object()


def _core_tokens(name: str) -> list[str]:
    tokens = name.casefold().translate(_PUNCT).split()
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    if len(tokens) >= 2 and tokens[-2] in _SUFFIXES and tokens[-1] in _SUFFIXES:
        del tokens[-2:]
    return [token for token in tokens if len(token) >= 4 and token.isascii()]


def _numbers(address: str) -> list[str]:
    numbers: list[str] = []
    for token in address.translate(_PUNCT).split():
        if token.isdigit():
            trimmed = token.lstrip("0")
            if trimmed:
                numbers.append(trimmed)
    return numbers


def _simhash(tokens: list[str]) -> int:
    acc = [0] * 64
    for token in tokens:
        value = hash(token) & ((1 << 64) - 1)
        for bit in range(64):
            acc[bit] += 1 if (value >> bit) & 1 else -1
    fingerprint = 0
    for bit, weight in enumerate(acc):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def blocking_keys(name: object, address: object) -> set[str]:
    """Return country-independent keys for one record."""
    name_text = "" if name is None else str(name)
    address_text = "" if address is None else str(address)
    tokens = _core_tokens(name_text)
    keys: set[str] = set()
    if any(ord(char) > 127 for char in name_text):
        skeleton = consonant_skeleton(name_text.casefold())
    else:
        chars: list[str] = []
        for token in tokens:
            for char in token:
                if char in "aeiouh":
                    continue
                chars.append("ks" if char == "x" else "k" if char == "c" else char)
        skeleton = "".join(chars)
    if len(skeleton) >= 5:
        keys.add("sk:" + skeleton[:10])
    compact = "".join(tokens)
    if len(compact) >= 6:
        keys.add("c6:" + compact[:8])
    for token in tokens:
        keys.add("t:" + token)
        keys.add("p4:" + token[:4])
    numbers = _numbers(address_text)
    for number in numbers:
        if len(number) >= 2:
            keys.add("n:" + number)
    if len(numbers) >= 2:
        keys.add("ns:" + "-".join(numbers[:4]))
    return keys


@dataclass
class BlockReport:
    true_links: int
    retrieved_uncapped: int
    retrieved_capped: int
    source1_scored: int
    median_shortlist: int
    dropped_common_keys: int
    macro_f05: float = 0.0
    singleton_accuracy: float = 0.0

    @property
    def uncapped_recall(self) -> float:
        return self.retrieved_uncapped / self.true_links if self.true_links else 0.0

    @property
    def capped_recall(self) -> float:
        return self.retrieved_capped / self.true_links if self.true_links else 0.0


def _dev_ids() -> set[str]:
    ids: set[str] = set()
    with SPLIT_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["dev_sample"] == "1":
                ids.add(row["source1_entity_id"])
    return ids


def _load_source1(dev_ids: set[str]) -> dict[str, tuple[str, set[str]]]:
    records: dict[str, tuple[str, set[str]]] = {}
    with config.TRAIN_SOURCE1_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entity_id = row["entity_id"]
            if entity_id not in dev_ids:
                continue
            country = row["country"].strip()
            records[entity_id] = (country, blocking_keys(row["business_name"], row["business_address"]))
            if len(records) == len(dev_ids):
                break
    return records


def _load_truth(dev_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {entity_id: set() for entity_id in dev_ids}
    with config.TRAIN_GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entity_id = row["source1_entity_id"]
            if entity_id not in truth:
                continue
            raw = row["matched_entity_ids"].strip()
            if raw:
                truth[entity_id].update(raw.split(","))
    return truth


def _index_candidates() -> tuple[dict[str, list[tuple[str, int]]], int]:
    postings: dict[str, list[tuple[str, int]] | object] = {}
    dropped = 0

    def add(key: str, entity_id: str, fingerprint: int) -> None:
        nonlocal dropped
        slot = postings.get(key)
        if slot is _TOO_COMMON:
            return
        if slot is None:
            postings[key] = [(entity_id, fingerprint)]
            return
        slot.append((entity_id, fingerprint))  # type: ignore[union-attr]
        limit = MAX_TOKEN_POSTING if "|t:" in key else MAX_POSTING
        if len(slot) > limit:  # type: ignore[arg-type]
            postings[key] = _TOO_COMMON
            dropped += 1

    for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        print(f"Indexing {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
                country = row["country"].strip()
                keys = blocking_keys(row["business_name"], row["business_address"])
                fingerprint = _simhash([key[2:] for key in keys if key.startswith("t:")])
                for key in keys:
                    if key.startswith("t:") or (key.startswith("n:") and len(key) >= 5):
                        add(country + "|" + key, row["entity_id"], fingerprint)
                if i % 1_000_000 == 0:
                    print(f"  {i:,} rows", flush=True)
    usable = {key: value for key, value in postings.items() if value is not _TOO_COMMON}
    return usable, dropped  # type: ignore[return-value]


def _entity_f05(labels: set[str], guess: set[str]) -> float:
    if not labels and not guess:
        return 1.0
    if not labels or not guess:
        return 0.0
    true_positives = len(labels & guess)
    if not true_positives:
        return 0.0
    precision = true_positives / len(guess)
    recall = true_positives / len(labels)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def measure_dev_recall() -> BlockReport:
    dev_ids = _dev_ids()
    print(f"Dev entities: {len(dev_ids):,}", flush=True)
    source1 = _load_source1(dev_ids)
    truth = _load_truth(dev_ids)
    postings, dropped = _index_candidates()
    print(f"Usable keys: {len(postings):,}  dropped common keys: {dropped:,}", flush=True)

    retrieved_uncapped = 0
    retrieved_capped = 0
    true_links = 0
    sizes: list[int] = []
    predicted: dict[str, set[str]] = {}
    singleton_count = 0
    rule_names = ("overlap", "two-supports", "top3", "top1", "name-top3")
    rule_scores = {name: 0.0 for name in rule_names}
    rule_singletons = {name: 0 for name in rule_names}
    for entity_id, (country, keys) in source1.items():
        labels = truth.get(entity_id, set())
        true_links += len(labels)
        counts: Counter[str] = Counter()
        token_buckets: list[set[str]] = []
        number_buckets: list[set[str]] = []
        fingerprints: dict[str, int] = {}
        for key in keys:
            bucket = postings.get(country + "|" + key, ())
            if not bucket:
                continue
            id_set: set[str] = set()
            for candidate_id, fingerprint in bucket:
                counts[candidate_id] += 1
                id_set.add(candidate_id)
                fingerprints[candidate_id] = fingerprint
            if key.startswith("t:") and len(key) >= 7:
                token_buckets.append(id_set)
            elif key.startswith("n:") and len(key) >= 5:
                number_buckets.append(id_set)
        support: Counter[str] = Counter()
        for token_ids in token_buckets:
            for number_ids in number_buckets:
                hit = token_ids & number_ids
                if 0 < len(hit) <= 8:
                    for candidate_id in hit:
                        support[candidate_id] += 1
        ranked = [candidate_id for candidate_id, _ in support.most_common(3)]
        source_fingerprint = _simhash([key[2:] for key in keys if key.startswith("t:")])
        close_ranked = [
            candidate_id for candidate_id, _ in support.most_common()
            if _hamming(source_fingerprint, fingerprints.get(candidate_id, 0)) <= 20
        ][:3]
        rule_guesses = {
            "overlap": set(support),
            "two-supports": {candidate_id for candidate_id, votes in support.items() if votes >= 2},
            "top3": set(ranked),
            "top1": set(ranked[:1]),
            "name-top3": set(close_ranked),
        }
        predicted[entity_id] = rule_guesses["top1"]
        for name, guess in rule_guesses.items():
            rule_scores[name] += _entity_f05(labels, guess)
            if not labels:
                rule_singletons[name] += not guess
        sizes.append(len(counts))
        retrieved_uncapped += sum(1 for match_id in labels if match_id in counts)
        if len(counts) > SHORTLIST:
            kept = {match_id for match_id, _ in counts.most_common(SHORTLIST)}
        else:
            kept = set(counts)
        retrieved_capped += sum(1 for match_id in labels if match_id in kept)
        if not labels:
            singleton_count += 1

    sizes.sort()
    scored = len(source1) or 1
    print("Rule sweep on dev:", flush=True)
    best_name = rule_names[0]
    best_f05 = -1.0
    for name in rule_names:
        macro = rule_scores[name] / scored
        singleton_accuracy = rule_singletons[name] / singleton_count if singleton_count else 0.0
        print(f"  {name}: macro F0.5 {macro:.4f}  singleton accuracy {singleton_accuracy:.4f}", flush=True)
        if macro > best_f05:
            best_name = name
            best_f05 = macro
    macro_f05 = best_f05
    singleton_accuracy = rule_singletons[best_name] / singleton_count if singleton_count else 0.0
    print(f"Best rule: {best_name}", flush=True)
    median = sizes[len(sizes) // 2] if sizes else 0
    return BlockReport(
        true_links, retrieved_uncapped, retrieved_capped, len(source1), median, dropped,
        macro_f05, singleton_accuracy,
    )


def main() -> int:
    report = measure_dev_recall()
    lines = [
        "DEV BLOCKING RECALL",
        f"Source 1 scored: {report.source1_scored:,}",
        f"True links: {report.true_links:,}",
        f"Uncapped pair-recall: {report.uncapped_recall:.4f} ({report.retrieved_uncapped:,})",
        f"Top-{SHORTLIST} pair-recall: {report.capped_recall:.4f} ({report.retrieved_capped:,})",
        f"Median candidate count before cap: {report.median_shortlist:,}",
        f"Keys dropped as too common (>{MAX_POSTING:,} rows): {report.dropped_common_keys:,}",
        f"Strict-rule macro F0.5: {report.macro_f05:.4f}",
        f"Singleton accuracy: {report.singleton_accuracy:.4f}",
    ]
    text = "\n".join(lines)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.REPORTS_DIR / "blocking_recall.txt"
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
