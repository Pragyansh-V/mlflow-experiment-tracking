# src/train.py

import mlflow
import mlflow.sklearn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

# Always resolve paths relative to project root, not wherever script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Always resolve paths relative to project root, not wherever script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

from data_loader import load_breast_cancer_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model(model_type: str, **kwargs) -> object:
    """Returns a sklearn model based on model_type string."""
    if model_type == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif model_type == "random_forest":
        return RandomForestClassifier(**kwargs)
    elif model_type == "svm":
        return SVC(probability=True, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: logistic_regression, random_forest, svm")


def save_confusion_matrix(y_test, y_pred, run_id: str) -> str:
    """Saves confusion matrix as a PNG and returns the file path."""
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_id[:8]}")

    path = f"artifacts/confusion_matrix_{run_id[:8]}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# ── Main Training Function ────────────────────────────────────────────────────

def train(model_type: str = "logistic_regression", **model_kwargs):
    """
    Trains a classifier on the Wisconsin Breast Cancer dataset.
    Logs params, metrics, and artifacts to MLflow.
    """

    # Load data
    X_train, X_test, y_train, y_test = load_breast_cancer_data()

    # Build pipeline — StandardScaler + model
    # Scaler is critical: SVM and LR are sensitive to feature scale
    model = build_model(model_type, **model_kwargs)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # ── MLflow Run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    mlflow.set_experiment("breast-cancer-classification")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n▶ MLflow Run ID: {run_id}")

        # Train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1_score":  f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_prob),
        }

        # Log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        for k, v in model_kwargs.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Log confusion matrix artifact
        cm_path = save_confusion_matrix(y_test, y_pred, run_id)
        mlflow.log_artifact(cm_path)

        # Print results
        print(f"\n{'─'*40}")
        print(f"  Model    : {model_type}")
        print(f"  Params   : {model_kwargs}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
        print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
        print(f"{'─'*40}\n")

        return run_id, metrics


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier with MLflow tracking")
    parser.add_argument("--model_type", type=str, default="logistic_regression",
                        choices=["logistic_regression", "random_forest", "svm"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    # Build model kwargs based on model type
    if args.model_type == "random_forest":
        kwargs = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "random_state": 42}
    elif args.model_type == "logistic_regression":
        kwargs = {"C": args.C, "max_iter": args.max_iter, "random_state": 42}
    elif args.model_type == "svm":
        kwargs = {"C": args.C}

    train(model_type=args.model_type, **kwargs)