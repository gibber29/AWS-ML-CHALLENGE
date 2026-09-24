"""Validated, string-preserving TSV loaders for challenge data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

ENTITY_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


class DataValidationError(ValueError):
    """Raised when an input file does not match the required schema."""


@dataclass(frozen=True)
class EntityResolutionData:
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: pd.DataFrame | None = None


def load_tsv(path: str | Path, required_columns: Iterable[str]) -> pd.DataFrame:
    """Load a TSV without converting empty strings or text-like IDs."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {input_path}")
    try:
        frame = pd.read_csv(
            input_path, sep="\t", dtype=str, keep_default_na=False,
            na_filter=False, encoding="utf-8",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read TSV {input_path}: {exc}") from exc
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{input_path} is missing required column(s) {missing}; "
            f"actual columns are {list(frame.columns)}"
        )
    return frame


def load_entity_file(path: str | Path) -> pd.DataFrame:
    return load_tsv(path, ENTITY_COLUMNS)


def load_ground_truth(path: str | Path = config.TRAIN_GROUND_TRUTH_PATH) -> pd.DataFrame:
    return load_tsv(path, GROUND_TRUTH_COLUMNS)


def load_training_data() -> EntityResolutionData:
    return EntityResolutionData(
        source1=load_entity_file(config.TRAIN_SOURCE1_PATH),
        source2=load_entity_file(config.TRAIN_SOURCE2_PATH),
        source3=load_entity_file(config.TRAIN_SOURCE3_PATH),
        ground_truth=load_ground_truth(),
    )


def load_test_data() -> EntityResolutionData:
    return EntityResolutionData(
        source1=load_entity_file(config.TEST_SOURCE1_PATH),
        source2=load_entity_file(config.TEST_SOURCE2_PATH),
        source3=load_entity_file(config.TEST_SOURCE3_PATH),
    )
