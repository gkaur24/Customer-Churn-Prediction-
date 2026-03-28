"""
Customer Churn Prediction - End-to-End Pipeline
================================================
Dataset: Telco Customer Churn (Kaggle)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Run:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn imbalanced-learn
    python churn_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw data and apply basic cleaning."""
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")

    # Fix TotalCharges (loaded as object due to blank strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Drop customerID — not predictive
    df.drop(columns=["customerID"], inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print(f"Cleaned shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame):
    """Generate key EDA plots and save to outputs/."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis – Customer Churn", fontsize=16, fontweight="bold")

    # 2a. Churn distribution
    churn_counts = df["Churn"].value_counts()
    axes[0, 0].bar(["No Churn", "Churned"], churn_counts.values, color=["#4CAF50", "#F44336"])
    axes[0, 0].set_title("Churn Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(churn_counts.values):
        axes[0, 0].text(i, v + 30, f"{v:,}\n({v/len(df):.1%})", ha="center")

    # 2b. Tenure vs Churn
    df.groupby("Churn")["tenure"].hist(ax=axes[0, 1], alpha=0.6, bins=30,
                                        color=["#4CAF50", "#F44336"], label=["No Churn", "Churned"])
    axes[0, 1].set_title("Tenure Distribution by Churn")
    axes[0, 1].set_xlabel("Tenure (months)")
    axes[0, 1].legend()

    # 2c. Monthly Charges vs Churn
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1, 0])
    axes[1, 0].set_title("Monthly Charges by Churn")
    axes[1, 0].set_xlabel("Churn (0=No, 1=Yes)")
    axes[1, 0].set_ylabel("Monthly Charges ($)")
    plt.sca(axes[1, 0])
    plt.title("Monthly Charges by Churn")

    # 2d. Contract type churn rate
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    contract_churn.plot(kind="bar", ax=axes[1, 1], color="#2196F3", edgecolor="white")
    axes[1, 1].set_title("Churn Rate by Contract Type")
    axes[1, 1].set_ylabel("Churn Rate")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=30)
    for i, v in enumerate(contract_churn.values):
        axes[1, 1].text(i, v + 0.005, f"{v:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("EDA plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful derived features."""
    df = df.copy()

    # Revenue-based features
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargePerService"] = df["MonthlyCharges"] / (
        df[["PhoneService", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"]]
        .apply(lambda col: (col != "No").astype(int))
        .sum(axis=1) + 1
    )

    # Tenure segmentation
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )

    # Service count (more services = more sticky)
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["ServiceCount"] = df[service_cols].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)

    # High-risk flag: month-to-month + fiber + high charges
    df["HighRiskFlag"] = (
        (df["Contract"] == "Month-to-month") &
        (df["InternetService"] == "Fiber optic") &
        (df["MonthlyCharges"] > df["MonthlyCharges"].median())
    ).astype(int)

    return df


def preprocess(df: pd.DataFrame):
    """Encode categoricals and scale numerics. Returns X, y, feature_names."""
    df = df.copy()

    # Separate target
    y = df.pop("Churn")

    # Encode all object/category columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    feature_names = df.columns.tolist()
    X = df.values
    return X, y.values, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELLING
# ══════════════════════════════════════════════════════════════════════════════

def build_models():
    """Return dict of model name → sklearn estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(
                                   n_estimators=300, learning_rate=0.05,
                                   max_depth=5, use_label_encoder=False,
                                   eval_metric="logloss", random_state=RANDOM_STATE
                               ),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test, feature_names):
    """Train, cross-validate, and compare all models. Return best model."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "model":    model,
            "cv_auc":   cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "test_auc": test_auc,
            "y_pred":   y_pred,
            "y_proba":  y_proba,
        }
        print(f"{name:25s} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
              f"| Test AUC: {test_auc:.4f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    aucs  = [results[n]["test_auc"] for n in names]
    bars  = ax.barh(names, aucs, color=["#90CAF9", "#A5D6A7", "#FFCC80"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Model Comparison")
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
                f"{auc:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_name = max(results, key=lambda n: results[n]["test_auc"])
    print(f"\nBest model: {best_name}")
    return results, best_name, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════

def deep_evaluate(best_result, best_name, X_test_sc, y_test):
    """Confusion matrix, classification report, ROC & PR curves."""
    y_pred  = best_result["y_pred"]
    y_proba = best_result["y_proba"]

    print(f"\n── {best_name} – Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churned"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{best_name} – Evaluation", fontsize=14, fontweight="bold")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churned"],
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[2])
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/best_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPLAINABILITY — SHAP
# ══════════════════════════════════════════════════════════════════════════════

def explain_with_shap(model, X_test_sc, feature_names):
    """SHAP summary & waterfall plots for explainability."""
    print("\nComputing SHAP values...")
    explainer = shap.Explainer(model, X_test_sc)
    shap_values = explainer(X_test_sc)

    # Summary (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_sc, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Importance (Impact on Churn Probability)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Single prediction waterfall (first churned customer)
    print("SHAP plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, scaler, feature_names):
    joblib.dump(model,        f"{OUTPUT_DIR}/best_model.pkl")
    joblib.dump(scaler,       f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_and_clean(DATA_PATH)

    # 2. EDA
    run_eda(df)

    # 3. Feature engineering
    df = engineer_features(df)
    X, y, feature_names = preprocess(df)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 5. Handle class imbalance with SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE – Train size: {len(X_train):,} | Churn rate: {y_train.mean():.1%}")

    # 6. Train & compare
    print("\n── Model Comparison ──")
    models = build_models()
    results, best_name, scaler = evaluate_models(
        models, X_train, X_test, y_train, y_test, feature_names
    )

    # 7. Deep evaluation
    X_test_sc = scaler.transform(X_test)
    deep_evaluate(results[best_name], best_name, X_test_sc, y_test)

    # 8. SHAP (only for tree models)
    best_model = results[best_name]["model"]
    if best_name in ("Random Forest", "XGBoost"):
        explain_with_shap(best_model, X_test_sc, feature_names)

    # 9. Save
    save_artifacts(best_model, scaler, feature_names)

    print("\nPipeline complete. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
