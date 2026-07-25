from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a bank customer will churn",
    version="1.0.0"
)

app.include_router(router)