# Telecom Customer Churn Prediction & Cloud MLOps Platform

## 1. Overview
Telecom operators lose millions annually to customer attrition. To tackle this high-stakes business problem, I built a production-grade, end-to-end MLOps solution: a machine learning classifier trained to predict subscriber churn, deployed via a serverless FastAPI backend on AWS Lambda and fronted by an interactive, public-facing Streamlit portfolio application. 

---

## 2. Why This Project
This project mirrors the core lifecycle responsibilities of a Machine Learning / MLOps Engineer:
- **Data Engineering & EDA:** Handling data cleaning, handling categorical variables and building feature alignment maps.
- **Model Experimentation & Training:** Training a binary classification model on telecom metrics (tenure, charges, contract types) to output reliable risk probabilities.
- **Cloud Architecture & Serverless Deployment:** Containerizing a FastAPI application and hosting it via AWS Lambda and Amazon API Gateway for scalable, low-cost execution.
- **Interactive Product Frontend:** Developing a user-friendly Streamlit web application to make the model accessible for real-time testing.
- **Automation & DevOps:** Setting up GitHub Actions CI/CD workflows and automated keep-alive pings to maintain continuous 24/7 availability.

---

## 3. Tech Stack
- **Languages & Core Libraries:** Python, Pandas, Scikit-learn, FastAPI, Mangum
- **Cloud & DevOps:** AWS Lambda, Amazon S3, Amazon API Gateway, Amazon ECR, AWS CodeBuild, Amazon CloudWatch, Docker, GitHub Actions, Swagger UI
- **Frontend & UI:** Streamlit, Requests, Streamlit Community Cloud

---

## 4. Data Source
The dataset used to train this model is the **Indian Telecom Customer Churn Prediction Dataset**, originally published by **Kiran Mehta** on Kaggle.
- **Dataset Link:** [Kiran Mehta1 on Kaggle](https://www.kaggle.com/datasets/kiranmehta1/indian-telecom-customer-churn-prediction-dataset?select=indian_customers_for_training.csv)
- **Data Overview:** Includes customer demographics, account tenure, subscribed communication/streaming services, contract types, payment methods and monthly/total billing charges.

---

## 5. Live Links & Portfolio Access
- **Interactive Web App Demo:** https://telecom-customer-churn-project-crcysyunzqsj5wdqhcmqtk.streamlit.app/ 
- **API Documentation (Swagger UI):** https://uvnhsaxyn9.execute-api.us-east-1.amazonaws.com/default/docs
- **GitHub Repository Source Code:** https://github.com/RuthKiarie/telecom-customer-churn-project

---

## 6. Architecture & Pipeline

The system is built on a serverless, containerized AWS architecture, integrated with automated CI/CD pipelines, an **HTTP API** (instead of REST API), and interactive API documentation:

```text
[ Developer / Git Commit ]
       │
       ▼
[ AWS CodeBuild ] ──(Builds Docker Image & Pushes)──> [ Amazon S3 / ECR Storage ]
       │                                                      │
       ▼                                                      ▼
[ GitHub Actions CI/CD ]                            [ AWS Lambda (FastAPI) ]
                                                              │
        ┌─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┐
        │                                                     │                                                     │
        ▼                                                     ▼                                                     ▼
[ IAM Roles & Policies ]                         [ Amazon CloudWatch ]                               [ Amazon API Gateway ]
(Least-Privilege Security)                       (Logging, Monitoring & Traces)                      (HTTP API Routing)
                                                                                                                    │
                                                                                                                    ├─► [ Swagger UI /docs ]
                                                                                                                    └─► [ Streamlit Frontend ]
