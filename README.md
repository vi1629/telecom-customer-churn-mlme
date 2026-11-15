# Telecom Customer Churn Prediction – ML Model Engineering

## 1. Project Overview

This project develops a machine learning system to classify telecom customers according to their recorded churn status using customer demographics, account information and subscribed services.

The project covers the machine learning lifecycle, including data preprocessing, model development, evaluation, serialization, API implementation, performance profiling and drift monitoring demonstration.

## 2. Dataset

The project uses the IBM Telco Customer Churn dataset.

- Total records: 7,043
- Total columns: 21
- Target variable: `Churn`
- Target classes: `Yes` and `No`
- Missing or blank `TotalCharges` values were handled during preprocessing.

The dataset represents a cross-sectional customer snapshot. Therefore, the task is framed as classification of recorded churn status rather than prospective next-cycle churn forecasting.

## 3. Methodology

The workflow includes:

1. Data loading and quality assessment
2. Missing-value handling and preprocessing
3. Exploratory data analysis
4. Stratified training, validation and test partitioning
5. Candidate model training and comparison
6. Decision-threshold selection using validation data
7. Final evaluation on the held-out test set
8. Model serialization using Joblib
9. REST API development using FastAPI
10. API performance profiling and drift monitoring demonstration

The candidate models include Logistic Regression, Random Forest and a Dummy baseline.

Logistic Regression was selected based on validation PR-AUC.

## 4. Final Model Performance

**Selected model:** Logistic Regression

**Decision threshold:** 0.5397

| Metric | Test Result |
|---|---:|
| Accuracy | 76.25% |
| Precision | 53.85% |
| Recall | 74.73% |
| F1-score | 62.59% |
| ROC-AUC | 84.46% |
| PR-AUC | 65.86% |

These results are based on the held-out test partition.

## 5. API Implementation

A REST API was developed using FastAPI.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks model availability |
| `/predict` | POST | Predicts churn for one customer |
| `/predict/batch` | POST | Predicts churn for multiple customers |

The API returns churn probability, predicted class and risk band for individual predictions.

### Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
uvicorn src.app:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 6. API Performance Profiling

The API was benchmarked using FastAPI TestClient in Google Colab.

| Measurement | Result |
|---|---:|
| Single prediction p50 latency | 24.93 ms |
| Single prediction p95 latency | 69.21 ms |
| Single prediction p99 latency | 84.67 ms |
| Mean latency | 31.71 ms |
| Single prediction throughput | 31.54 requests/sec |
| Batch throughput (32 records) | 1,115.81 records/sec |

These measurements represent in-process endpoint performance and exclude external network transport latency.

## 7. Drift Monitoring

An offline distribution comparison was performed between the training and held-out test datasets.

Population Stability Index (PSI) and statistical distribution checks were used to demonstrate the drift monitoring approach.

This is an offline evaluation demonstration and does not represent continuous production monitoring.

## 8. Repository Structure

```text
telecom-customer-churn-mlme/
│
├── artifacts/
│   ├── churn_pipeline.joblib
│   ├── model_metadata.json
│   └── README.md
│
├── notebooks/
│   ├── Telecom_Customer_Churn_MLME.ipynb
│   └── README.md
│
├── reports/
│   ├── run_summary.json
│   ├── api_profiling.json
│   ├── api_latency_samples.csv
│   ├── drift_monitoring_demo.csv
│   └── README.md
│
├── results/
│   ├── validation_model_comparison.csv
│   ├── heldout_test_metrics.csv
│   ├── data_split_summary.csv
│   └── README.md
│
├── src/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## 9. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- FastAPI
- Pydantic
- Google Colab

## 10. Project Repository

GitHub: https://github.com/vi1629/telecom-customer-churn-mlme
