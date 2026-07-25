from fastapi import APIRouter

from app.schemas import CustomerData
from src.predict import predict_customer

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "Running"
    }


@router.post("/predict")
def predict(data: CustomerData):

    result = predict_customer(data.model_dump())

    return result