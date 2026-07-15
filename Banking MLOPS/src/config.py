from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "bank-additional.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
TARGET_COLUMN = "y"
REQUESTED_TARGET_COLUMN = "Y"

NUMERIC_COLUMNS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def resolve_target_column(columns) -> str:
    """Return the actual target column, accepting user-requested Y or dataset y."""
    if REQUESTED_TARGET_COLUMN in columns:
        return REQUESTED_TARGET_COLUMN
    if TARGET_COLUMN in columns:
        return TARGET_COLUMN
    raise ValueError(f"Target column not found. Expected '{REQUESTED_TARGET_COLUMN}' or '{TARGET_COLUMN}'.")
