from fastapi import FastAPI

from app.api.routes.predict import router as predict_router

app = FastAPI(
    title="Dissanayake POS ML Engine",
    version="0.1.0",
    description="Demand forecasting API for Dissanayake Super Web POS",
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(predict_router, prefix="/api/v1", tags=["prediction"])
