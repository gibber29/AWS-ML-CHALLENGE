"""See which key families true dev links actually share, by country."""

import csv
import random
from collections import Counter

from . import config
from .retrieve_v2 import record_keys
from .split import SPLIT_PATH

SAMPLE = 400


def main() -> int:
    rng = random.Random(2026)
    picked: dict[str, list[tuple[str, str]]] = {"US": [], "India": []}
    pools: dict[str, list[tuple[str, str]]] = {"US": [], "India": []}
    countries = {}
    with SPLIT_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["dev_sample"] == "1":
                countries[row["source1_entity_id"]] = row["country"]
    with config.TRAIN_GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entity_id = row["source1_entity_id"]
            country = countries.get(entity_id)
            raw = row["matched_entity_ids"].strip()
            if country not in pools or not raw:
                continue
            pools[country].append((entity_id, raw.split(",")[0]))
    for country, pairs in pools.items():
        rng.shuffle(pairs)
        picked[country] = pairs[:SAMPLE]

    wanted = {entity_id for pairs in picked.values() for pair in pairs for entity_id in pair}
    text: dict[str, tuple[str, str]] = {}
    for path in (config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        print(f"Reading {path.name}", flush=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in wanted:
                    text[row["entity_id"]] = (row["business_name"], row["business_address"])
                    if len(text) == len(wanted):
                        break
        if len(text) == len(wanted):
            break

    lines = ["TRUE-LINK KEY OVERLAP", f"Sample per country: {SAMPLE}"]
    families = ("t", "p", "s", "sk", "c", "n", "ns", "aw")
    for country, pairs in picked.items():
        shared = Counter()
        none = 0
        missing_address = 0
        for left_id, right_id in pairs:
            if left_id not in text or right_id not in text:
                continue
            left_name, left_address = text[left_id]
            right_name, right_address = text[right_id]
            if not left_address.strip() or not right_address.strip():
                missing_address += 1
            left = {key.split(":", 1)[0] for key, _w in record_keys(left_name, left_address)}
            right = {key.split(":", 1)[0] for key, _w in record_keys(right_name, right_address)}
            # Family overlap is not enough: the actual key values must match.
            left_keys = {key for key, _w in record_keys(left_name, left_address)}
            right_keys = {key for key, _w in record_keys(right_name, right_address)}
            both = {key.split(":", 1)[0] for key in left_keys & right_keys}
            if not both:
                none += 1
            for family in both:
                shared[family] += 1
            del left, right
        lines.append(f"{country}: no shared key {none}/{len(pairs)}  one side missing address {missing_address}/{len(pairs)}")
        for family in families:
            lines.append(f"  {family}: {shared[family]}/{len(pairs)}")
    text_out = "\n".join(lines)
    output = config.REPORTS_DIR / "link_key_overlap.txt"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(text_out + "\n", encoding="utf-8")
    print(text_out, flush=True)
    print(f"Saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
