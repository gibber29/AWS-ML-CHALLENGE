import tempfile
from pathlib import Path
import unittest

import pandas as pd

from business_entity_resolution.src.data_loader import (
    DataValidationError,
    load_entity_file,
)
from business_entity_resolution.src.evaluate import (
    EvaluationInputError,
    entity_f05,
    evaluate_frames,
    parse_match_ids,
)
from business_entity_resolution.src.normalize import (
    fold_accents,
    normalize_address,
    normalize_name,
    normalize_text,
)


class DataLoaderTests(unittest.TestCase):
    def test_tsv_loading_preserves_strings_and_empty_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entities.tsv"
            path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-001\tAcme, Inc.\t\tUS\n",
                encoding="utf-8",
            )
            frame = load_entity_file(path)
        self.assertEqual(frame.loc[0, "entity_id"], "S1-001")
        self.assertEqual(frame.loc[0, "business_address"], "")
        self.assertEqual(frame.loc[0, "business_name"], "Acme, Inc.")

    def test_missing_required_column_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tsv"
            path.write_text("entity_id\tbusiness_name\nS1-1\tA\n", encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "missing required column"):
                load_entity_file(path)


class EvaluationTests(unittest.TestCase):
    def test_ground_truth_parser(self):
        self.assertEqual(parse_match_ids("S2-1,S3-2"), {"S2-1", "S3-2"})
        self.assertEqual(parse_match_ids(""), set())
        with self.assertRaises(EvaluationInputError):
            parse_match_ids("S1-1")

    def test_f05_calculation(self):
        actual = entity_f05({"S2-1", "S3-2"}, {"S2-1", "S2-9", "S3-2"})
        self.assertAlmostEqual(actual, 5 / 7)

    def test_singleton_behavior(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"S2-1"}), 0.0)
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)

    def test_duplicate_prediction_ids_are_rejected(self):
        with self.assertRaisesRegex(EvaluationInputError, "Duplicate IDs"):
            parse_match_ids("S2-1,S2-1")

    def test_macro_not_global_and_country_diagnostics(self):
        truth = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1,S3-1", ""],
        })
        predictions = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1", ""],
        })
        report = evaluate_frames(truth, predictions, {"S1-1": "US", "S1-2": "India"})
        self.assertAlmostEqual(report.macro_f05, (5 / 6 + 1) / 2)
        self.assertEqual(report.singleton_accuracy, 1.0)
        self.assertAlmostEqual(report.country_scores["US"], 5 / 6)
        self.assertEqual(report.country_scores["India"], 1.0)


class NormalizationTests(unittest.TestCase):
    def test_unicode_accents_ampersand_and_boundaries(self):
        self.assertEqual(normalize_text(" Café & Fils42, S.A.S. "), "café and fils 42 s a s")
        self.assertEqual(fold_accents("École Française"), "ecole francaise")

    def test_name_preserves_suffix_and_exposes_core(self):
        value = normalize_name("Example & Sons Pvt. Ltd.")
        self.assertEqual(value.raw, "Example & Sons Pvt. Ltd.")
        self.assertEqual(value.normalized, "example and sons pvt ltd")
        self.assertEqual(value.core, "example and sons")

    def test_address_tokens_numbers_and_empty_fields(self):
        address = normalize_address("12B Rue de l'École")
        self.assertIn("12", address.numeric_tokens)
        self.assertIn("école", address.tokens)
        self.assertEqual(normalize_name("").normalized, "")
        self.assertEqual(normalize_address(None).tokens, ())


if __name__ == "__main__":
    unittest.main()
