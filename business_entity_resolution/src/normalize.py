"""Conservative, Unicode-safe representations for entity matching."""

from dataclasses import dataclass
import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_LETTER_NUMBER_BOUNDARY_RE = re.compile(
    r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", re.UNICODE
)
_NUMBER_RE = re.compile(r"\d+(?:[./-]\d+)*", re.UNICODE)
_LEGAL_SUFFIXES = tuple(
    tuple(item.split())
    for item in (
        "private limited", "public limited", "limited liability partnership",
        "limited liability company", "pvt ltd", "pte ltd", "incorporated",
        "corporation", "company", "limited", "llc", "llp", "plc", "corp",
        "inc", "ltd", "co", "sarl", "sasu", "sas", "sci", "eurl", "sa",
        "gmbh",
    )
)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def normalize_text(value: object) -> str:
    """NFKC-normalize, casefold and normalize punctuation/spacing."""
    text = unicodedata.normalize("NFKC", _text(value)).casefold().replace("&", " and ")
    chars = [" " if unicodedata.category(c)[0] in {"P", "Z"} else c for c in text]
    text = _LETTER_NUMBER_BOUNDARY_RE.sub(" ", "".join(chars))
    return _SPACE_RE.sub(" ", text).strip()


def fold_accents(value: object) -> str:
    """Return a separate accent-folded form while preserving non-Latin scripts."""
    decomposed = unicodedata.normalize("NFKD", normalize_text(value))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _remove_legal_suffixes(name: str) -> str:
    tokens = name.split()
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            size = len(suffix)
            if len(tokens) > size and tuple(tokens[-size:]) == suffix:
                del tokens[-size:]
                changed = True
                break
    return " ".join(tokens)


@dataclass(frozen=True)
class NormalizedName:
    raw: str
    normalized: str
    accent_folded: str
    core: str
    core_accent_folded: str


@dataclass(frozen=True)
class NormalizedAddress:
    raw: str
    normalized: str
    accent_folded: str
    tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]


def normalize_name(value: object) -> NormalizedName:
    raw = _text(value)
    normalized = normalize_text(raw)
    core = _remove_legal_suffixes(normalized)
    return NormalizedName(raw, normalized, fold_accents(normalized), core, fold_accents(core))


def normalize_address(value: object) -> NormalizedAddress:
    raw = _text(value)
    normalized = normalize_text(raw)
    return NormalizedAddress(
        raw, normalized, fold_accents(normalized), tuple(normalized.split()),
        tuple(_NUMBER_RE.findall(normalized)),
    )
