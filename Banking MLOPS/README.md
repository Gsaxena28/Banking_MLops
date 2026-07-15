# Banking MLOps Logistic Regression Pipeline

Complete Python MLOps pipeline for the Bank Marketing dataset using logistic regression.

## What is included

- Pandas-based data ingestion from `data/raw/bank-additional.csv`
- Great Expectations data validation
- DVC dataset and pipeline versioning
- NumPy/Pandas/scikit-learn preprocessing and logistic regression training
- MLflow experiment tracking and model registration
- FastAPI `/predict` endpoint served by Uvicorn
- Unit tests with pytest
- Dockerfile for the API service
- GitHub Actions CI/CD workflow for tests, validation, training, and DockerHub push

## Project structure

```text
.
├── api/main.py
├── data/raw/bank-additional.csv
├── data/raw/bank-additional.csv.dvc
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_validation.py
│   └── train.py
└── tests/
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Validate data

```bash
python -m src.data_validation
```

## Train model

```bash
python -m src.train
```

The trained artifact is saved to `models/logistic_regression_model.pkl`.

## Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "job": "blue-collar",
    "marital": "married",
    "education": "basic.9y",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "fri",
    "duration": 487,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1
  }'
```

## DVC commands

```bash
dvc status
dvc repro
```

If you configure remote storage later:

```bash
dvc remote add -d storage <remote-url>
dvc push
```

## MLflow

Training logs parameters, metrics, the pickle artifact, and an MLflow model. The script attempts to register the model as `BankMarketingLogisticRegression`.

```bash
mlflow ui
```

Then open `http://localhost:5000`.
