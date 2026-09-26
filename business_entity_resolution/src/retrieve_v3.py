"""Experimental additive retrieval views and inexpensive address-aware ranking.

Original keys remain available. Extra keys have country document-frequency caps;
no transliteration package, external lookup, or globally unrestricted gram index.
Weights/budgets are experiment settings, not fitted claims of improvement.
"""

from dataclasses import dataclass
from itertools import combinations
import re
import unicodedata

from .normalize import consonant_skeleton, normalize_name, fold_accents, normalize_text, _remove_legal_suffixes
from .retrieve_v2 import _ADDR_STOP, _cap, record_keys, unique_keys

VERSION = "v3-folded-address-skeleton-1"
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_ABBREVIATIONS = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
                  "ln": "lane", "dr": "drive", "ct": "court", "fl": "floor"}
_STOP = _ADDR_STOP | {"street", "avenue", "boulevard", "district", "delhi", "mumbai", "hyderabad"}


def address_parts(address):
    words = _WORDS.findall(fold_accents(address))
    words = [_ABBREVIATIONS.get(word, word) for word in words]
    numbers = frozenset(word.lstrip("0") or "0" for word in words if word.isdigit())
    content = frozenset(word for word in words if len(word) >= 4 and not word.isdigit() and word not in _STOP)
    return content, numbers


def enhanced_keys(name, address):
    keys = record_keys(name, address)
    core = _remove_legal_suffixes(normalize_text(name))
    folded_name = core if core.isascii() else "".join(
        c for c in unicodedata.normalize("NFKD", core) if not unicodedata.combining(c))
    # Both accented and unaccented records emit the folded namespace.
    for token in folded_name.split():
        if len(token) >= 3:
            keys.append(("f:t:" + token, 1.5))
    if len(folded_name) >= 6:
        keys.append(("f:c:" + folded_name.replace(" ", "")[:16], 2.8))
    words, numbers = address_parts(address)
    keys.extend(("f:aw:" + word, .8) for word in words)
    words = sorted(words, key=lambda w: (-len(w), w))[:6]
    numbers = sorted((n for n in numbers if len(n) >= 2), key=lambda n: (-len(n), n))[:4]
    for left, right in combinations(sorted(words), 2):
        keys.append((f"a2:{left}|{right}", 2.2))
    for word in words:
        for number in numbers:
            if len(number) >= 3:
                keys.append((f"an:{word}|{number}", 2.8))
    for left, right in combinations(sorted(numbers), 2):
        keys.append((f"n2:{left}|{right}", 2.5))
    skeleton = consonant_skeleton(core)[:40]
    # Local consonant grams tolerate leading/trailing word and script changes.
    if len(skeleton) >= 5:
        keys.extend(("sg:" + skeleton[i:i+4], .65) for i in range(len(skeleton)-3))
    return unique_keys(keys)


def enhanced_cap(key):
    if key.startswith("f:"):
        return _cap(key[2:])
    family = key.split(":", 1)[0]
    return {"a2": 2500, "an": 2500, "n2": 2500, "sg": 1200}.get(family, _cap(key))


def grams(text):
    compact = "".join(c for c in text if c.isalnum())
    return frozenset(compact[i:i+3] for i in range(max(1, len(compact)-2))) if compact else frozenset()


@dataclass(frozen=True)
class PairView:
    name: str
    tokens: frozenset
    name_grams: frozenset
    skeleton_grams: frozenset
    address: frozenset
    numbers: frozenset


def pair_view(name, address):
    normalized = normalize_name(name)
    words, numbers = address_parts(address)
    return PairView(normalized.core_accent_folded, frozenset(normalized.core_accent_folded.split()),
                    grams(normalized.core_accent_folded), grams(consonant_skeleton(normalized.core)),
                    words, numbers)


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 0.


def dice(left, right):
    return 2 * len(left & right) / (len(left) + len(right)) if left or right else 0.


def pair_features(left, right):
    shared = left.numbers & right.numbers
    return {
        "name_exact": float(bool(left.name) and left.name == right.name),
        "name_tokens": jaccard(left.tokens, right.tokens),
        "name_chars": dice(left.name_grams, right.name_grams),
        "name_skeleton": dice(left.skeleton_grams, right.skeleton_grams),
        "address_words": jaccard(left.address, right.address),
        "address_containment": len(left.address & right.address) / max(1, min(len(left.address), len(right.address))),
        "number_long": float(any(len(n) >= 4 for n in shared)),
        "number_medium": float(any(len(n) == 3 for n in shared)),
        "number_short": float(bool(shared)),
        "number_conflict": float(bool(left.numbers and right.numbers and not shared)),
        "missing_address": float(not (left.address or left.numbers) or not (right.address or right.numbers)),
    }


def pair_rank(features, use_address=True):
    f = features
    name = max(f["name_tokens"], f["name_chars"], .85 * f["name_skeleton"])
    score = 2.5 * name + .5 * f["name_exact"]
    if use_address:
        address = .9 * f["address_words"] + .3 * f["address_containment"]
        number = .65 * f["number_long"] + .4 * f["number_medium"] + .15 * f["number_short"]
        score += address + number + .5 * name * address - .3 * f["number_conflict"]
    return score


def route_pool(ranked, routes, limit=1000):
    """Reserve equal experiment budgets for distinct name/address routes."""
    if limit < 4:
        return ranked[:limit]
    families = ({"t", "c", "f_t", "f_c"}, {"p", "s", "sk", "sg"},
                {"aw", "f_aw", "a2", "an", "n", "ns", "n2"})
    chosen = set(ranked[:limit // 2])
    budget = (limit - limit // 2) // len(families)
    for family in families:
        candidates = (candidate for candidate in ranked if routes[candidate] & family and candidate not in chosen)
        for _, candidate in zip(range(budget), candidates):
            chosen.add(candidate)
    for candidate in ranked:
        if len(chosen) >= limit:
            break
        chosen.add(candidate)
    # Retain deterministic original IDF ordering within the selected union.
    return [candidate for candidate in ranked if candidate in chosen]
