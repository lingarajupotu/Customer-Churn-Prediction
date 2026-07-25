import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """
    Load trained model
    """
    return joblib.load(MODEL_PATH)


def predict_customer(customer_data):
    """
    Predict customer churn
    """

    model = load_model()

    input_df = pd.DataFrame([customer_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        label = "Customer Will Churn"
    else:
        label = "Customer Will Not Churn"

    return {
        "prediction": label,
        "churn_probability": f"{probability*100:.2f}%"
    }
    
#result = predict_customer(sample_customer)

#print(result)