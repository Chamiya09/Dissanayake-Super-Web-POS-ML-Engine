from fastapi import APIRouter

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import PredictorService

router = APIRouter()
predictor_service = PredictorService()


@router.post("/predict", response_model=PredictionResponse)
def predict_demand(payload: PredictionRequest) -> PredictionResponse:
    predicted_quantity = predictor_service.predict(payload)
    return PredictionResponse(
        product_id=payload.product_id,
        forecast_date=payload.forecast_date,
        predicted_quantity=predicted_quantity,
        model_version=predictor_service.model_version,
    )
