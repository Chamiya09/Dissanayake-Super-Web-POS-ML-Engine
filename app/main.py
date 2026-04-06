from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("ml_engine")
logging.basicConfig(level=logging.INFO)


class PredictionRequest(BaseModel):
    Item_ID: str = Field(..., min_length=1, description="Item identifier or product name")
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    rolling_mean_4_weeks: float
    SellingPrice: float
    Month: int = Field(..., ge=1, le=12)
    WeekOfYear: int = Field(..., ge=1, le=53)


class PredictionResponse(BaseModel):
    predicted_quantity: int


def _resolve_model_path() -> Path:
    # app/main.py -> project root -> models/xgboost_champion.pkl
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models" / "xgboost_champion.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded artifact does not provide a predict() method.")

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "Loaded model does not expose feature_names_in_. "
            "Train the model using a pandas DataFrame so inference columns can be aligned safely."
        )

    app.state.model = model
    app.state.expected_columns = list(model.feature_names_in_)

    logger.info("Model loaded from %s", model_path)
    logger.info("Expected feature count: %d", len(app.state.expected_columns))
    yield


app = FastAPI(
    title="Dissanayake POS ML Engine",
    version="1.0.0",
    description="Demand forecasting API for Dissanayake Super Web POS",
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    model = getattr(app.state, "model", None)
    expected_columns: list[str] | None = getattr(app.state, "expected_columns", None)

    if model is None or expected_columns is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        payload: dict[str, Any] = request.model_dump()

        # Build inference frame from one request.
        inference_df = pd.DataFrame([payload])

        # One-hot encode Item_ID exactly like training-time logic.
        inference_df = pd.get_dummies(inference_df, columns=["Item_ID"], prefix="item", dtype=int)

        # Align columns to training schema and fill unseen/missing dummies with zeros.
        inference_df = inference_df.reindex(columns=expected_columns, fill_value=0)

        raw_prediction = model.predict(inference_df)
        raw_prediction_value = float(raw_prediction[0])

        # Model was trained on log1p(target), so reverse via expm1.
        predicted_quantity = float(np.expm1(raw_prediction_value))
        predicted_quantity = max(0.0, predicted_quantity)

        return PredictionResponse(predicted_quantity=int(round(predicted_quantity)))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
