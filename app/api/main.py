"""
Run the API server from the project root:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for demand forecasting requests."""

    Item_ID: str | int = Field(..., description="Product identifier")
    lag_7: float = Field(..., description="Quantity sold 7 days ago")
    lag_14: float = Field(..., description="Quantity sold 14 days ago")
    Is_Weekend: bool = Field(..., description="True if forecast date is on weekend")
    SellingPrice: float = Field(..., gt=0, description="Current selling price")
    day_of_week: int = Field(..., ge=0, le=6, description="Day index where Monday=0")
    month: int = Field(..., ge=1, le=12, description="Month number from forecast date")


def _resolve_model_path() -> Path:
    """Resolve model path relative to project root."""
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "xgboost_champion.pkl"
    return model_path


def _get_payload_dict(payload: PredictionRequest) -> dict[str, Any]:
    """Compatibility helper for both Pydantic v1 and v2."""
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory once at startup and keep it in app state."""
    model_path = _resolve_model_path()

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. Train and export the model before starting the API."
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

    app.state.model = model
    app.state.model_path = str(model_path)
    app.state.model_features = list(getattr(model, "feature_names_in_", []))
    yield


app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0.0",
    description="XGBoost-powered demand prediction service",
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", tags=["prediction"])
def predict(payload: PredictionRequest) -> dict[str, float]:
    """Predict demand quantity from lag and contextual features."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        payload_dict = _get_payload_dict(payload)

        # Use model-declared training feature order when available.
        expected_features = list(getattr(model, "feature_names_in_", []))
        if not expected_features:
            expected_features = ["lag_7", "lag_14", "day_of_week", "month"]

        missing_features = [feature for feature in expected_features if feature not in payload_dict]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required model features: {missing_features}",
            )

        model_input = pd.DataFrame([{feature: payload_dict[feature] for feature in expected_features}])
        prediction = model.predict(model_input)
        predicted_quantity = float(prediction[0])

        return {"predicted_quantity": predicted_quantity}

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input values: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
