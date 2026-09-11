import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("India_Telecom_Churn_Prediction")

def train():
    df = pd.read_csv("indian_customers_for_training.csv").dropna()
    
    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
    y = df["Churn"].map({'Yes': 1, 'No': 0, '1': 1, '0': 0}).fillna(df["Churn"])

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)

        # Save feature columns as a dictionary inside the MLflow artifact
        joblib.dump(feature_columns, "feature_columns.pkl")
        mlflow.log_artifact("feature_columns.pkl", artifact_path="churn_model")
        
        mlflow.sklearn.log_model(rf, name="churn_model")
        print("Training complete! Model and feature names saved.")

if __name__ == "__main__":
    train()