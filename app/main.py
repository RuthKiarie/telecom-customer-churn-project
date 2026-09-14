from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import skops.io as sio
from mangum import Mangum
import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

model = None
feature_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_columns
    try:
        base_dir = Path(__file__).resolve().parent
        
        # DEBUG: Print everything currently sitting in the app folder inside the container
        print(f"DEBUG: Contents of {base_dir}:")
        if base_dir.exists():
            print(os.listdir(base_dir))
        else:
            print("CRITICAL: base_dir itself does not exist!")

        model_path = base_dir / "churn_model.skops"
        feature_path = base_dir / "feature_columns.pkl"

        if model_path.exists():
            # Get untrusted types dynamically and pass them as trusted for our own model file
            unknown_types = sio.get_untrusted_types(file=model_path)
            model = sio.load(model_path, trusted=unknown_types)
            print("Skops model loaded successfully!")
        else:
            print(f"CRITICAL: Model file NOT found at {model_path}")

        if feature_path.exists():
            feature_columns = joblib.load(feature_path)
            print("Feature columns loaded successfully!")
        else:
            print(f"CRITICAL: Feature columns file NOT found at {feature_path}")

    except Exception as e:
        print(f"Error loading model during startup: {e}")
    
    yield
    
    model = None
    feature_columns = None


app = FastAPI(
    title="India Telecom Customer Churn Prediction API",
    description="REST API serving real-time predictions for customer churn using a trained model.",
    version="2.0.0",
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

handler = Mangum(app)