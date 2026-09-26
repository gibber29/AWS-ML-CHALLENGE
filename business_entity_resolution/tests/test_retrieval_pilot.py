import tempfile
from pathlib import Path
import unittest

from business_entity_resolution.src.retrieval_pilot import (
    ProjectedIndex, classify, measure, select_pilot, scan_projected,
)
from business_entity_resolution.src.retrieve_v2 import (
    record_keys, legacy_record_keys, pack_id, index_rows, retrieve,
)


def row(entity_id, name, address="", country="US"):
    return dict(entity_id=entity_id, business_name=name, business_address=address, country=country)


class PilotTests(unittest.TestCase):
    def test_projection_matches_full_repaired_index(self):
        query = row("S1-1", "Acme Bakery", "0141 Longstreet")
        records = [row("S2-1", "Acme Acme", "141 Longstreet Longstreet"),
                   row("S3-2", "Bakery", "141 Longstreet"), row("S2-3", "Other", "99 Unknown"),
                   row("S3-4", "Acme Bakery", "141 Longstreet", "India")]
        keys = record_keys(query["business_name"], query["business_address"])
        index = ProjectedIndex({"US|" + k for k, _ in keys})
        for record in records:
            index.add(record)
        full, counts, countries = index_rows([tuple(r.values()) for r in records])
        expected = retrieve("US", keys, full, counts, countries["US"], 100)
        ranked, _, _ = index.rank(query)
        self.assertEqual([pack_id(x) for x in expected], ranked)
        self.assertEqual(index.df, {k: counts[k] for k in index.df})
        without_routes = index.rank(query, with_routes=False)
        self.assertEqual(index.rank(query)[:2], without_routes[:2])
        self.assertEqual(without_routes[2], {})

    def test_legacy_duplicate_boost_and_corrected_tie(self):
        query = row("S1-1", "", "Longstreet")
        index = ProjectedIndex({"US|aw:longstreet"})
        for record in [row("S2-2", "", "Longstreet Longstreet"),
                       row("S2-1", "", "Longstreet"),
                       *[row(f"S3-{i}", "Other") for i in range(10)]]:
            index.add(record)
        legacy, scores_before, _ = index.rank(query, legacy=True)
        repaired, scores_after, _ = index.rank(query)
        self.assertEqual(legacy[0], pack_id("S2-2"))
        self.assertEqual(repaired[0], pack_id("S2-1"))
        self.assertAlmostEqual(scores_before[pack_id("S2-2")], 2 * scores_before[pack_id("S2-1")])
        self.assertEqual(scores_after[pack_id("S2-2")], scores_after[pack_id("S2-1")])

    def test_caps_use_occurrences_only_for_legacy(self):
        index = ProjectedIndex({"US|aw:longstreet"})
        index.caps["US|aw:longstreet"] = 2
        index.add(row("S2-1", "", "Longstreet Longstreet"))
        index.add(row("S2-2", "", "Longstreet"))
        self.assertFalse(index.usable("US|aw:longstreet", legacy=True))
        self.assertTrue(index.usable("US|aw:longstreet"))
        index.add(row("S2-3", "", "Longstreet"))
        self.assertFalse(index.usable("US|aw:longstreet"))
        self.assertNotIn("US|aw:longstreet", index.postings)

    def test_reason_precedence(self):
        self.assertEqual(classify(set(), set(), None, 100), "NO_RAW_KEY")
        self.assertEqual(classify({"t:a"}, set(), None, 100), "ALL_SHARED_KEYS_PRUNED")
        self.assertEqual(classify({"t:a"}, {"t:a"}, 101, 100), "RANKED_BELOW_K")
        self.assertEqual(classify({"t:a"}, {"t:a"}, 100, 100), "IN_TOP_K")

    def test_cache_roundtrip_and_stale_rejection(self):
        index = ProjectedIndex({"US|aw:longstreet"})
        index.add(row("S2-1", "", "Longstreet"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            index.save(path, {"version": "fixture"}, {})
            loaded, _ = ProjectedIndex.load(path, {"version": "fixture"})
            self.assertEqual(loaded.postings, index.postings)
            with self.assertRaises(ValueError):
                ProjectedIndex.load(path, {"version": "different"})

    def test_all_siblings_and_singleton_oracle(self):
        records = [row("S2-1", "Acme Bakery"), row("S3-2", "Acme Bakery")]
        queries = {"S1-1": row("S1-1", "Acme Bakery"), "S1-2": row("S1-2", "Solo")}
        keys = {"US|" + k for r in queries.values() for k, _ in record_keys(r["business_name"], "")}
        index = ProjectedIndex(keys)
        for record in records:
            index.add(record)
        with tempfile.TemporaryDirectory() as directory:
            report = measure(index, queries, {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()},
                             {r["entity_id"]: r for r in records}, Path(directory) / "loss.tsv")
        overall = report["groups"]["all"]
        self.assertEqual(overall["true_links"], 2)
        self.assertEqual(overall["singletons"], 1)
        self.assertEqual(overall["at_k"]["20"]["oracle_macro_f05"], 1.)

    def test_pilot_determinism_and_size(self):
        source = {f"S1-{i}": row(f"S1-{i}", "Common", country="India" if i % 2 else "US") for i in range(50)}
        truth = {key: set() if i % 3 else {"S2-1", "S3-2"} for i, key in enumerate(source)}
        a, strata = select_pilot(source, truth, 20, 2026)
        b, _ = select_pilot(dict(reversed(list(source.items()))), truth, 20, 2026)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertEqual(sum(s["selected"] for s in strata), 20)

    def test_parallel_counting_matches_sequential(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tsv"
            path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                            "S2-1\tAcme Acme\tLongstreet Longstreet\tUS\n"
                            "S3-2\tAcme\tLongstreet\tUS\n"
                            "S3-3\tOther\t\tIndia\n", encoding="utf-8")
            sequential = ProjectedIndex({"US|aw:longstreet", "US|t:acme"})
            parallel = ProjectedIndex(set(sequential.df))
            a = scan_projected(sequential, [path], {"S3-2"}, workers=1)
            b = scan_projected(parallel, [path], {"S3-2"}, workers=2, batch_size=1)
            self.assertEqual(a, b)
            self.assertEqual(sequential.df, parallel.df)
            self.assertEqual(sequential.tf, parallel.tf)
            self.assertEqual(sequential.postings, parallel.postings)
            self.assertEqual(sequential.country_n, parallel.country_n)


if __name__ == "__main__":
    unittest.main()
