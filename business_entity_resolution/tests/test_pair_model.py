import unittest
import csv
from pathlib import Path
import tempfile
import numpy as np

from business_entity_resolution.src.pair_model import evaluate, feature_vector, write_predictions, FEATURES
from business_entity_resolution.src.retrieve_v3 import pair_view
from business_entity_resolution.src.retrieve_v2 import pack_id


class PairModelTests(unittest.TestCase):
    def test_entity_macro_singletons_and_unretrieved_truth(self):
        data = dict(y=np.array([1, 0, 0, 1]), offsets=np.array([0, 2, 3, 4]),
                    truth_counts=np.array([2, 0, 1]), countries=np.array(["US", "India", "US"]))
        report = evaluate(data, np.array([.9, .8, .1, .6]), .7)
        self.assertAlmostEqual(report["macro_f05"], .5)
        self.assertAlmostEqual(report["pair_precision"], .5)
        self.assertAlmostEqual(report["pair_recall"], 1 / 3)
        self.assertEqual(report["singleton_accuracy"], 1)
        self.assertAlmostEqual(report["oracle_macro_f05"], (5/6 + 1 + 1) / 3)

    def test_empty_candidate_lists_still_evaluate_every_entity(self):
        data = dict(y=np.array([]), offsets=np.array([0, 0, 0]),
                    truth_counts=np.array([0, 1]), countries=np.array(["US", "US"]))
        report = evaluate(data, np.array([]), .5)
        self.assertEqual(report["entities"], 2)
        self.assertEqual(report["macro_f05"], .5)
        self.assertEqual(report["predicted_empty_rate"], 1)

    def test_ids_are_not_features_except_source(self):
        left = pair_view("Acme", "0141 Market Road")
        right = pair_view("Acme LLC", "141 Market Rd")
        a = feature_vector(left, right, .8, {"t", "aw", "n"}, pack_id("S2-1"))
        b = feature_vector(left, right, .8, {"t", "aw", "n"}, pack_id("S2-999"))
        c = feature_vector(left, right, .8, {"t", "aw", "n"}, pack_id("S3-999"))
        self.assertEqual(a, b)
        self.assertEqual(a[:-1], c[:-1])
        self.assertEqual(len(a), len(FEATURES))
        self.assertNotEqual(a[-1], c[-1])

    def test_prediction_writer_keeps_empty_rows_and_does_not_cap_siblings(self):
        data = dict(ids=np.array(["S1-1", "S1-2"]), offsets=np.array([0, 4, 4]),
                    candidates=np.array([pack_id(f"S2-{i}") for i in range(1, 5)]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_predictions(data, np.ones(4), .5, output)
            with (output / "confirmation_matching.tsv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["matched_entity_ids"], "S2-1,S2-2,S2-3,S2-4")
        self.assertEqual(records[1]["matched_entity_ids"], "")


if __name__ == "__main__":
    unittest.main()
