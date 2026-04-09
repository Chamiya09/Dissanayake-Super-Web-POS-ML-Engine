from datetime import date
import logging
import traceback

from fastapi import APIRouter, HTTPException, Query

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import PredictorService

router = APIRouter()
predictor_service = PredictorService()
logger = logging.getLogger("ml_predict_route")


def _log_detailed_exception(context: str, exc: Exception) -> None:
    """Print developer-friendly traceback with exact failing frame."""
    print(f"[{context}] Exception type: {type(exc).__name__}")
    print(f"[{context}] Exception message: {exc}")
    print(f"[{context}] Full traceback:\n{traceback.format_exc()}")

    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        last = tb[-1]
        print(f"[{context}] Failing line: {last.filename}:{last.lineno} in {last.name}")
        print(f"[{context}] Cause line text: {last.line}")

    logger.exception("%s failed", context)


@router.post("/predict", response_model=PredictionResponse)
def predict_demand(payload: PredictionRequest) -> PredictionResponse:
    try:
        predicted_quantity = predictor_service.predict(payload)
        return PredictionResponse(
            product_id=payload.product_id,
            forecast_date=payload.forecast_date,
            predicted_quantity=predicted_quantity,
            model_version=predictor_service.model_version,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        _log_detailed_exception("predict_demand", exc)
        raise HTTPException(status_code=400, detail="Invalid prediction input.") from exc
    except Exception as exc:
        _log_detailed_exception("predict_demand", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@router.get("/api/forecast")
def get_forecast(
    product_id: str = Query(..., min_length=1, description="Product identifier"),
    timeframe: str = Query(..., description="weekly or monthly"),
) -> dict[str, object]:
    normalized_timeframe = str(timeframe).strip().lower()
    if normalized_timeframe not in {"weekly", "monthly"}:
        raise HTTPException(status_code=400, detail="timeframe must be 'weekly' or 'monthly'.")

    normalized_product_id = product_id.strip()
    if not normalized_product_id:
        raise HTTPException(status_code=400, detail="product_id is required.")

    try:
        # Placeholder inference path for route-level error handling demonstration.
        request = PredictionRequest(product_id=normalized_product_id, forecast_date=date.today())
        predicted_quantity = predictor_service.predict(request)

        if predicted_quantity is None:
            raise HTTPException(
                status_code=404,
                detail="Product ID not found or lacks sufficient historical data to forecast.",
            )

        return {
            "product_id": normalized_product_id,
            "timeframe": normalized_timeframe,
            "predicted_demand": int(round(float(predicted_quantity))),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        _log_detailed_exception("get_forecast", exc)
        raise HTTPException(status_code=400, detail="Invalid forecast input.") from exc
    except Exception as exc:
        _log_detailed_exception("get_forecast", exc)
        raise HTTPException(status_code=500, detail="Forecast generation failed.") from exc
