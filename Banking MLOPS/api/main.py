from __future__ import annotations

import pickle
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURE_COLUMNS, MODEL_PATH

app = FastAPI(title="Bank Marketing Logistic Regression API", version="1.0.0")

model = None


class BankCustomer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

    model_config = {"populate_by_name": True}


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run `python -m src.train` first.")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: BankCustomer) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    payload = customer.model_dump(by_alias=True)
    row = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return {
        "prediction": "yes" if prediction == 1 else "no",
        "prediction_label": prediction,
        "positive_probability": probability,
    }
