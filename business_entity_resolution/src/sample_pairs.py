"""Write a small, local sample of true matches and same-country lookalikes.

Raw text stays in reports/, which is gitignored. The sample is for choosing
blocking keys, not for scoring.
"""

from collections import defaultdict
import csv
import random

from . import config
from .normalize import normalize_address, normalize_name
from .split import SPLIT_PATH

OUT_PATH = config.REPORTS_DIR / "pair_examples.txt"
PER_COUNTRY = 15


def _load_dev_ids() -> set[str]:
    ids: set[str] = set()
    with SPLIT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["dev_sample"] == "1":
                ids.add(row["source1_entity_id"])
    return ids


def _wanted_matches(dev_ids: set[str]) -> dict[str, list[str]]:
    rng = random.Random(config.RANDOM_SEED)
    by_country: dict[str, list[tuple[str, list[str]]]] = {"US": [], "India": []}
    with config.TRAIN_GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = row["source1_entity_id"]
            if entity_id not in dev_ids:
                continue
            raw = row["matched_entity_ids"].strip()
            if not raw:
                continue
            matches = raw.split(",")
            by_country_key = None
            # Country is filled after the source-1 scan. Keep every linked dev row
            # and assign country in the next pass. Store under a placeholder.
            by_country.setdefault("_", []).append((entity_id, matches))
            del by_country_key
    # The placeholder bucket is the full linked dev set. Country filtering happens
    # once source 1 records are loaded.
    return {entity_id: matches for entity_id, matches in by_country["_"]}


def main() -> int:
    dev_ids = _load_dev_ids()
    matches = _wanted_matches(dev_ids)
    rng = random.Random(config.RANDOM_SEED)

    records: dict[str, tuple[str, str, str]] = {}
    wanted = set(matches)
    for path in (config.TRAIN_SOURCE1_PATH, config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                entity_id = row["entity_id"]
                if entity_id in wanted or entity_id.startswith("S2-") or entity_id.startswith("S3-"):
                    if entity_id in wanted or entity_id.startswith(("S2-", "S3-")):
                        pass
                if entity_id in wanted:
                    records[entity_id] = (row["business_name"], row["business_address"], row["country"])
                    wanted.discard(entity_id)
                    if not wanted and path == config.TRAIN_SOURCE1_PATH:
                        break

    # Source 1 pass above only keeps dev matches' source-1 rows if they were in
    # `wanted` at the start. Rebuild wanted as all match targets, then rescan 2 and 3.
    targets: set[str] = set()
    chosen: dict[str, list[tuple[str, list[str]]]] = {"US": [], "India": []}
    pool = list(matches.items())
    rng.shuffle(pool)
    for entity_id, match_ids in pool:
        record = records.get(entity_id)
        if record is None:
            continue
        country = record[2]
        if country in chosen and len(chosen[country]) < PER_COUNTRY:
            chosen[country].append((entity_id, match_ids[:2]))
            targets.update(match_ids[:2])
        if all(len(rows) >= PER_COUNTRY for rows in chosen.values()):
            break

    if targets:
        found: dict[str, tuple[str, str, str]] = {}
        for path in (config.TRAIN_SOURCE2_PATH, config.TRAIN_SOURCE3_PATH):
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    entity_id = row["entity_id"]
                    if entity_id in targets:
                        found[entity_id] = (row["business_name"], row["business_address"], row["country"])
                        if len(found) == len(targets):
                            break
            if len(found) == len(targets):
                break
        records.update(found)

    lines = ["TRUE MATCHES", ""]
    for country, rows in chosen.items():
        lines.append(f"## {country}")
        for entity_id, match_ids in rows:
            left = records[entity_id]
            left_name = normalize_name(left[0])
            left_address = normalize_address(left[1])
            lines.append(f"S1 {entity_id} | {left[0]} | {left[1]}")
            lines.append(f"   core={left_name.core!r} digits={list(left_address.numeric_tokens)}")
            for match_id in match_ids:
                right = records.get(match_id)
                if right is None:
                    lines.append(f"  MISSING {match_id}")
                    continue
                right_name = normalize_name(right[0])
                right_address = normalize_address(right[1])
                lines.append(f"  {match_id} | {right[0]} | {right[1]}")
                lines.append(f"   core={right_name.core!r} digits={list(right_address.numeric_tokens)}")
            lines.append("")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {sum(len(rows) for rows in chosen.values())} source-1 examples to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
