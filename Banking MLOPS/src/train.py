from __future__ import annotations

import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, DATA_PATH, FEATURE_COLUMNS, MODEL_DIR, MODEL_PATH, NUMERIC_COLUMNS, resolve_target_column
from src.data_validation import validate_with_great_expectations


def load_training_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    validate_with_great_expectations(df)
    target_column = resolve_target_column(df.columns)
    x = df[FEATURE_COLUMNS].copy()
    y = df[target_column].map({"no": 0, "yes": 1})
    return x, y


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model() -> dict[str, float]:
    x, y = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()

    mlflow.set_experiment("banking-logistic-regression")
    with mlflow.start_run(run_name="logistic-regression-training"):
        params = {
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "test_size": 0.2,
            "random_state": 42,
        }
        mlflow.log_params(params)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_test, predictions), "classification_report.txt")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open("wb") as file:
            pickle.dump(model, file)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="BankMarketingLogisticRegression",
            input_example=x_test.head(3),
        )

    print(f"Model saved to {MODEL_PATH}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    train_model()
