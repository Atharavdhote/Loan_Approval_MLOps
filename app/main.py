
import joblib
import pandas as pd

from fastapi import FastAPI
from app.schemas import LoanApplication

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML-powered Loan Approval Prediction System",
    version="1.0.0"
)

# Load saved artifacts
model = joblib.load("models/loan_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_loan(application: LoanApplication):

    data = pd.DataFrame([application.model_dump()])

    # Handle Dependents value
    data["Dependents"] = data["Dependents"].replace("3+", "3")

    # Apply preprocessing
    processed_data = preprocessor.transform(data)

    # Predict
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "loan_status": result,
        "prediction": int(prediction),
        "approval_probability": round(float(probability), 4)
    }
