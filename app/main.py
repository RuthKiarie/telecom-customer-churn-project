from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from mangum import Mangum

model = None
feature_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_columns
    try:
        # AWS Lambda sets LAMBDA_TASK_ROOT to /var/task
        # Fall back to base path relative to main.py for local testing
        task_root = os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Look in task root first, then fallback to current directory
        model_path = os.path.join(task_root, "churn_model.pkl")
        feature_path = os.path.join(task_root, "feature_columns.pkl")

        if not os.path.exists(model_path):
            model_path = "churn_model.pkl"
        if not os.path.exists(feature_path):
            feature_path = "feature_columns.pkl"

        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        print(f"Attempting to load features from: {os.path.abspath(feature_path)}")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Model file NOT found at {model_path}")

        if os.path.exists(feature_path):
            feature_columns = joblib.load(feature_path)
            print("Feature columns loaded successfully!")
        else:
            print(f"Feature columns file NOT found at {feature_path}")

    except Exception as e:
        print(f"Error loading model during startup: {e}")
    
    yield
    
    model = None
    feature_columns = None

app = FastAPI(
    title="India Telecom Customer Churn Prediction API",
    description="REST API serving real-time predictions for customer churn using a trained model.",
    version="1.0.0",
    root_path="/default",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "India Telecom Churn Prediction API",
        "docs_url": "/docs"
    }

class ChurnRequest(BaseModel):
    features: dict[str, float | int | str]

@app.post("/predict")
def predict_churn(payload: ChurnRequest):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model or feature schema not loaded.")

    try:
        df_raw = pd.DataFrame([payload.features])
        df_encoded = pd.get_dummies(df_raw, drop_first=True)
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df_aligned)
        probabilities = model.predict_proba(df_aligned) if hasattr(model, "predict_proba") else None

        churn_result = int(prediction[0])
        churn_prob = float(probabilities[0][1]) if probabilities is not None else None

        return {
            "churn_prediction": churn_result,
            "churn_status": "Yes" if churn_result == 1 else "No",
            "probability_churn": churn_prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

handler = Mangum(app, api_gateway_base_path="/default") 