import requests
import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Dashboard", page_icon="📊", layout="centered"
)

st.title("📊 Telecom Churn Predictor")
st.write(
    "Use this dashboard to evaluate customer churn risk in real-time."
)

# Sidebar or input fields for customer features
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
  gender = st.selectbox("Gender", ["Female", "Male"])
  senior_citizen = st.selectbox("Senior Citizen", [0, 1])
  partner = st.selectbox("Partner", ["Yes", "No"])
  dependents = st.selectbox("Dependents", ["Yes", "No"])
  tenure = st.slider("Tenure (Months)", 1, 72, 12)
  phone_service = st.selectbox("Phone Service", ["Yes", "No"])
  multiple_lines = st.selectbox(
      "Multiple Lines", ["No", "Yes", "No phone service"]
  )
  internet_service = st.selectbox(
      "Internet Service", ["DSL", "Fiber optic", "No"]
  )
  online_security = st.selectbox(
      "Online Security", ["No", "Yes", "No internet service"]
  )
  online_backup = st.selectbox(
      "Online Backup", ["No", "Yes", "No internet service"]
  )

with col2:
  device_protection = st.selectbox(
      "Device Protection", ["No", "Yes", "No internet service"]
  )
  tech_support = st.selectbox(
      "Tech Support", ["No", "Yes", "No internet service"]
  )
  streaming_tv = st.selectbox(
      "Streaming TV", ["No", "Yes", "No internet service"]
  )
  streaming_movies = st.selectbox(
      "Streaming Movies", ["No", "Yes", "No internet service"]
  )
  contract = st.selectbox(
      "Contract Type", ["Month-to-month", "One year", "Two year"]
  )
  paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
  payment_method = st.selectbox(
      "Payment Method",
      [
          "Electronic check",
          "Mailed check",
          "Bank transfer (automatic)",
          "Credit card (automatic)",
      ],
  )
  monthly_charges = st.number_input("Monthly Charges ($)", value=70.35)
  total_charges = st.number_input("Total Charges ($)", value=840.20)

# Predict Button
if st.button("Predict Churn Risk", type="primary"):
  # Construct payload matching your FastAPI schema
  payload = {
      "features": {
          "gender": gender,
          "SeniorCitizen": senior_citizen,
          "Partner": partner,
          "Dependents": dependents,
          "tenure": tenure,
          "PhoneService": phone_service,
          "MultipleLines": multiple_lines,
          "InternetService": internet_service,
          "OnlineSecurity": online_security,
          "OnlineBackup": online_backup,
          "DeviceProtection": device_protection,
          "TechSupport": tech_support,
          "StreamingTV": streaming_tv,
          "StreamingMovies": streaming_movies,
          "Contract": contract,
          "PaperlessBilling": paperless_billing,
          "PaymentMethod": payment_method,
          "MonthlyCharges": monthly_charges,
          "TotalCharges": total_charges,
      }
  }

  with st.spinner("Calling AWS Lambda API..."):
    try:
      # Send request to your live API Gateway endpoint
      response = requests.post(
          "https://uvnhsaxyn9.execute-api.us-east-1.amazonaws.com/default/predict",
          json=payload,
      )

      if response.status_code == 200:
        result = response.json()
        churn_status = result.get("churn_status")
        prob = result.get("probability_churn", 0.0)

        st.markdown("---")
        st.subheader("Prediction Results")
        if churn_status == "Yes":
          st.error(
              f"⚠️ **High Churn Risk!**\n\nProbability of Churn: **{prob:.2%}**"
          )
        else:
          st.success(
              f"✅ **Low Churn Risk.**\n\nProbability of Churn: **{prob:.2%}**"
          )
      else:
        st.error(
            f"API Error: Received status code {response.status_code} -"
            f" {response.text}"
        )
    except Exception as e:
      st.error(f"Failed to connect to API: {e}")