
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "churn_pipeline.joblib"
META_PATH = BASE_DIR / "artifacts" / "model_metadata.json"

app = FastAPI(title="Telecom Churn Prediction API", version="1.1.0")
model = None
metadata = None


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: str | None = None
    Dependents: str | None = None
    tenure: float | None = None
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float | None = None
    TotalCharges: float | None = None


@app.on_event("startup")
def load_artifacts():
    global model, metadata
    if not MODEL_PATH.exists() or not META_PATH.exists():
        model = None
        metadata = None
        return
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text())


@app.get("/health")
def health():
    ready = model is not None and metadata is not None
    return {
        "status": "ok" if ready else "unavailable",
        "model_loaded": ready
    }


@app.post("/predict")
def predict(record: CustomerRecord):
    if model is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model artefact is unavailable"
        )

    row = pd.DataFrame([record.model_dump()])
    row = row[metadata["feature_columns"]]

    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata["decision_threshold"])
    prediction = int(probability >= threshold)

    band = (
        "High" if probability >= 0.60
        else ("Medium" if probability >= 0.30 else "Low")
    )

    return {
        "churn_probability": probability,
        "predicted_churn": prediction,
        "predicted_label": "Yes" if prediction else "No",
        "risk_band": band,
        "threshold": threshold
    }


@app.post("/predict/batch")
def predict_batch(records: list[CustomerRecord]):
    if model is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model artefact is unavailable"
        )

    if not records:
        raise HTTPException(
            status_code=422,
            detail="At least one record is required"
        )

    frame = pd.DataFrame(
        [record.model_dump() for record in records]
    )
    frame = frame[metadata["feature_columns"]]

    probabilities = model.predict_proba(frame)[:, 1]
    threshold = float(metadata["decision_threshold"])

    return {
        "predictions": [
            {
                "churn_probability": float(probability),
                "predicted_churn": int(probability >= threshold)
            }
            for probability in probabilities
        ]
    }
