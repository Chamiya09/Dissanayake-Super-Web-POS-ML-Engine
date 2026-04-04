import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv("DATABASE_URL", "")
    model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")


settings = Settings()
