# src/evaluate.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

from data_loader import load_breast_cancer_data

# ── Config ────────────────────────────────────────────────────────────────────

TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
MODEL_URI    = "models:/breast-cancer-classifier@production"


def load_production_model():
    """Loads the @production model from MLflow Registry."""
    print(f"\n Loading production model...")
    print(f"  URI: {MODEL_URI}")
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"  Model loaded: {type(model.named_steps['classifier']).__name__}")
    return model


def save_roc_curve(y_test, y_prob, run_id: str) -> str:
    """Saves ROC curve as PNG artifact."""
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="crimson")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Production Model")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(artifacts_dir, f"roc_curve_{run_id[:8]}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_confusion_matrix(y_test, y_pred, run_id: str) -> str:
    """Saves confusion matrix as PNG artifact."""
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Production Model Evaluation")

    path = os.path.join(artifacts_dir, f"eval_confusion_matrix_{run_id[:8]}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def evaluate():
    """
    Loads @production model, runs evaluation on held-out test set,
    logs results as a new MLflow run under 'breast-cancer-evaluation'.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load data and model
    _, X_test, _, y_test = load_breast_cancer_data()
    model = load_production_model()

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    # Log to a separate evaluation experiment
    mlflow.set_experiment("breast-cancer-evaluation")

    with mlflow.start_run(run_name="production-model-eval") as run:
        run_id = run.info.run_id

        # Log model reference
        mlflow.log_param("model_uri", MODEL_URI)
        mlflow.log_param("evaluated_on", "held-out test set (20%)")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts
        cm_path  = save_confusion_matrix(y_test, y_pred, run_id)
        roc_path = save_roc_curve(y_test, y_prob, run_id)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        # Print full report
        print(f"\n{'═'*50}")
        print(f"  Production Model Evaluation")
        print(f"{'═'*50}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
        print(f"{'═'*50}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=["Benign", "Malignant"]))
        print(f"  MLflow Run: {run_id}")
        print(f"  Artifacts : confusion matrix + ROC curve logged\n")

    return metrics


if __name__ == "__main__":
    evaluate()