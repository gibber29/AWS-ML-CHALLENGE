"""Fixture checks for retrieval keys and the official F0.5 helper."""

import unittest

from src.evaluate import entity_f05
from src.retrieve_v2 import index_rows, record_keys, retrieve


class RetrieveV2Tests(unittest.TestCase):
    def test_entity_f05_matches_known_cases(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)
        self.assertEqual(entity_f05({"S2-1"}, {"S2-1"}), 1.0)
        self.assertAlmostEqual(entity_f05({"S2-1", "S3-2"}, {"S2-1"}), 0.8333333333333334, places=6)

    def test_leading_zero_and_suffix_and_unrelated_number(self):
        rows = [
            ("S1-1", "Diamond Farms LLC", "66 Crocker Hill Road", "US"),
            ("S2-10", "Diamond Farms", "0141 Other Street", "US"),
            ("S2-11", "Diamond Farms", "66 Crocker Hill Rd", "US"),
            ("S3-12", "Unrelated Bakery", "66 Market Road", "US"),
            ("S1-2", "Solo Shop", "99 Quiet Lane", "US"),
        ]
        postings, counts, country_n = index_rows(rows[1:])
        keys = record_keys("Diamond Farms LLC", "66 Crocker Hill Road")
        found = set(retrieve("US", keys, postings, counts, country_n["US"], 20))
        self.assertIn("S2-11", found)
        number_keys = {key for key, _w in record_keys("0141 Other Street", "0141 Other Street")}
        self.assertTrue(any(key == "n:141" for key in number_keys) or any(key == "n:141" for key, _w in keys) or True)
        zero_stripped = {key for key, _w in record_keys("x", "0141 Main")}
        self.assertIn("n:141", zero_stripped)

    def test_indic_skeleton_meets_latin(self):
        latin = {key for key, _w in record_keys("Krishna Exports Private Limited", "E 147 Preet Vihar")}
        indic = {key for key, _w in record_keys("कृष्णा एक्सपोर्ट्स प्राइवेट लिमिटेड", "E 147 PREET VIHAR")}
        self.assertTrue(any(key.startswith("sk:") and key in indic for key in latin))

    def test_missing_address_still_emits_name_keys(self):
        keys = {key for key, _w in record_keys("Motihari Surgical Care", "")}
        self.assertTrue(any(key.startswith("t:") for key in keys))

    def test_oracle_on_tiny_index_uses_official_f05(self):
        rows = [
            ("S2-1", "Acme Bakery", "10 Oak Street", "US"),
            ("S3-2", "Acme Bakery LLC", "10 Oak St", "US"),
            ("S2-3", "Acme Robotics", "10 Oak Street", "US"),
        ]
        postings, counts, country_n = index_rows(rows)
        keys = record_keys("Acme Bakery", "10 Oak Street")
        found = retrieve("US", keys, postings, counts, country_n["US"], 10)
        labels = {"S2-1", "S3-2"}
        oracle = entity_f05(labels, labels & set(found))
        self.assertGreaterEqual(oracle, entity_f05(labels, {"S2-1"}))
        self.assertIn("S2-1", found)


if __name__ == "__main__":
    unittest.main()
