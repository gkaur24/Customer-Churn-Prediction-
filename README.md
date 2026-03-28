# 🔄 Customer Churn Prediction — End-to-End Data Science Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io)

## Overview

End-to-end machine learning pipeline to predict customer churn for a telecom company.
The model identifies at-risk customers so retention teams can intervene proactively — reducing churn rate and improving customer lifetime value.

**Key Result:** XGBoost model achieved **ROC-AUC of 0.85** on holdout test set, outperforming the logistic regression baseline by 6 percentage points.

---

## Business Problem

Customer acquisition costs 5–7× more than retention. A 5% reduction in churn can increase profits by 25–95% (Harvard Business Review). This model enables the business to:

- Rank all customers by churn probability monthly
- Flag high-risk customers for proactive outreach
- Understand *why* a customer is at risk (via SHAP)
- Prioritise retention budget toward customers most likely to respond

---

## Project Structure

```
churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (from Kaggle)
├── outputs/
│   ├── eda.png                                  # EDA visualisations
│   ├── model_comparison.png                     # Model comparison chart
│   ├── evaluation.png                           # ROC, PR, confusion matrix
│   ├── shap_summary.png                         # SHAP feature importance
│   ├── best_model.pkl                           # Saved model
│   └── scaler.pkl                               # Saved scaler
├── churn_pipeline.py      # Full training pipeline (run this)
├── churn_notebook.ipynb   # Annotated walkthrough notebook
├── app.py                 # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data & EDA
- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features
- Target: `Churn` (Yes/No) — **~26% churn rate** (class imbalance addressed with SMOTE)
- Key findings from EDA:
  - Month-to-month contracts churn at **43%** vs 11% for 1-year contracts
  - Customers with tenure < 12 months are highest risk
  - Fiber optic + high monthly charges = strong churn signal

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `AvgMonthlyCharge` | TotalCharges / tenure — lifetime value proxy |
| `ServiceCount` | Number of add-on services subscribed |
| `TenureGroup` | Binned tenure: 0–1yr, 1–2yr, 2–4yr, 4+yr |
| `HighRiskFlag` | Month-to-month + Fiber + high charges |

### 3. Modelling
Three models compared with 5-fold stratified cross-validation:

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.793 | 0.791 |
| Random Forest | 0.836 | 0.841 |
| **XGBoost** | **0.851** | **0.853** |

### 4. Evaluation
- Primary metric: **ROC-AUC** (ranking quality)
- Also tracked: Precision, Recall, F1 for churned class
- Optimised threshold for business use case (higher recall = fewer missed churners)

### 5. Explainability
SHAP (SHapley Additive exPlanations) used to:
- Rank global feature importance
- Explain individual predictions to retention agents
- Identify actionable intervention points

Top SHAP features: `Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `TechSupport`

---

## Key Insights & Business Recommendations

| Risk Factor | Churn Rate | Recommended Action |
|---|---|---|
| Month-to-month contract | 43% | Offer discount to switch to annual |
| Tenure < 12 months | High | 30/90 day onboarding check-in call |
| Fiber optic + high charges | High | Bundle loyalty discount |
| Electronic check payment | High | Nudge to auto-pay (+small incentive) |
| No tech support | Higher | Free 3-month trial for at-risk customers |

---

## Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place in `data/`.

### Run pipeline
```bash
python churn_pipeline.py
```

### Launch Streamlit app
```bash
streamlit run app.py
```

---

## Requirements
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.43
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.28
joblib>=1.3
```

---

