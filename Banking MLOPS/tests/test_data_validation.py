import pandas as pd

from src.config import DATA_PATH, FEATURE_COLUMNS, resolve_target_column


def test_dataset_has_expected_columns():
    df = pd.read_csv(DATA_PATH)
    target_column = resolve_target_column(df.columns)
    assert set(FEATURE_COLUMNS + [target_column]) == set(df.columns)


def test_target_contains_binary_labels():
    df = pd.read_csv(DATA_PATH)
    target_column = resolve_target_column(df.columns)
    assert set(df[target_column].unique()).issubset({"yes", "no"})
