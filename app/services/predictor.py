from app.schemas.prediction import PredictionRequest


class PredictorService:
    def __init__(self, model_version: str = "baseline-v1") -> None:
        self.model_version = model_version

    def predict(self, request: PredictionRequest) -> float:
        # Placeholder: replace with real model inference using loaded .pkl artifact
        return 1.0
