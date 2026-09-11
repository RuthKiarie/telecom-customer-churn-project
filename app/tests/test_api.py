import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root directory is top of sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

from app.main import app

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"

def test_predict_churn_endpoint():
    sample_payload = {
        "features": {
            "Age": 34,
            "Tenure": 12,
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0
        }
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 200
        json_data = response.json()
        assert "churn_prediction" in json_data
        assert "churn_status" in json_data
        assert "probability_churn" in json_data