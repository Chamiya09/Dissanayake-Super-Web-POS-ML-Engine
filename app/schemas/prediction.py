from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    product_id: str = Field(..., description="Product identifier")
    forecast_date: date = Field(..., description="Date to forecast")


class PredictionResponse(BaseModel):
    product_id: str
    forecast_date: date
    predicted_quantity: float
    model_version: str
