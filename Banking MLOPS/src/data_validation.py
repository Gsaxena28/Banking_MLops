from __future__ import annotations

import pandas as pd

from src.config import CATEGORICAL_COLUMNS, DATA_PATH, FEATURE_COLUMNS, NUMERIC_COLUMNS, resolve_target_column


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_with_great_expectations(df: pd.DataFrame) -> bool:
    """Validate schema, missing values, uniqueness, and basic quality rules."""
    try:
        import great_expectations as gx
    except ImportError as exc:
        raise RuntimeError("great_expectations is required. Run: pip install -r requirements.txt") from exc

    target_column = resolve_target_column(df.columns)

    # GE 0.18 supports from_pandas; some newer versions keep it under dataset.
    try:
        validator = gx.from_pandas(df)
    except AttributeError:
        from great_expectations.dataset import PandasDataset

        validator = PandasDataset(df)

    expected_columns = FEATURE_COLUMNS + [target_column]
    validator.expect_table_columns_to_match_set(expected_columns, exact_match=True)

    for column in expected_columns:
        validator.expect_column_values_to_not_be_null(column)

    for column in NUMERIC_COLUMNS:
        validator.expect_column_values_to_be_of_type(column, "int64" if column in {"age", "duration", "campaign", "pdays", "previous"} else "float64")

    for column in CATEGORICAL_COLUMNS:
        validator.expect_column_values_to_be_of_type(column, "object")

    validator.expect_column_values_to_be_between("age", min_value=17, max_value=100)
    validator.expect_column_values_to_be_between("campaign", min_value=1)
    validator.expect_column_values_to_be_between("previous", min_value=0)
    validator.expect_column_distinct_values_to_be_in_set(target_column, ["yes", "no"])

    result = validator.validate()
    if not result["success"]:
        failed = [
            item["expectation_config"]["expectation_type"]
            for item in result["results"]
            if not item["success"]
        ]
        raise ValueError(f"Great Expectations validation failed: {failed}")

    return True


def main() -> None:
    df = load_data()
    validate_with_great_expectations(df)
    print(f"Data validation passed for {DATA_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    main()
