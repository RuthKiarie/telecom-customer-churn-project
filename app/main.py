from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import mlflow
import os

model = None
feature_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_columns
    try:
        experiment = mlflow.get_experiment_by_name("India_Telecom_Churn_Prediction")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            if not runs.empty:
                latest_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{latest_run_id}/churn_model"
                
                model = mlflow.sklearn.load_model(model_uri)
                client = mlflow.tracking.MlflowClient()
                feature_path = client.download_artifacts(latest_run_id, "churn_model/feature_columns.pkl")
                feature_columns = joblib.load(feature_path)
                print("Model and Feature Columns loaded successfully!")
    except Exception as e:
        print(f"Error loading model during startup: {e}")
    
    yield
    
    # Cleanup on shutdown if needed
    model = None
    feature_columns = None

app = FastAPI(
    title="India Telecom Customer Churn Prediction API",
    description="REST API serving real-time predictions for customer churn using an MLflow-tracked model.",
    version="1.0.0",
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