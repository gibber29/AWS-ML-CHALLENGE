"""Project-wide paths and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "business_entity_resolution"
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

TRAIN_FILES = {
    "train_source1": TRAIN_SOURCE1_PATH,
    "train_source2": TRAIN_SOURCE2_PATH,
    "train_source3": TRAIN_SOURCE3_PATH,
    "train_ground_truth": TRAIN_GROUND_TRUTH_PATH,
}
TEST_FILES = {
    "test_source1": TEST_SOURCE1_PATH,
    "test_source2": TEST_SOURCE2_PATH,
    "test_source3": TEST_SOURCE3_PATH,
}
RANDOM_SEED = 2026
