# src/register_model.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mlflow
from mlflow.tracking import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────────

TRACKING_URI   = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_NAME      = "breast-cancer-classifier"

# The best run we identified from the sweep
BEST_PARAMS = {
    "model_type": "logistic_regression",
    "C": "0.1",
    "max_iter": "1000",
}


def find_best_run(client: MlflowClient, experiment_name: str) -> str:
    """
    Finds the best run by ROC AUC in the given experiment.
    Returns the run_id.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment.")

    best_run = runs[0]
    print(f"\n Best run found:")
    print(f"  Run ID    : {best_run.info.run_id}")
    print(f"  Model     : {best_run.data.params.get('model_type')}")
    print(f"  C         : {best_run.data.params.get('C')}")
    print(f"  ROC AUC   : {best_run.data.metrics.get('roc_auc'):.4f}")
    print(f"  F1 Score  : {best_run.data.metrics.get('f1_score'):.4f}")
    print(f"  Recall    : {best_run.data.metrics.get('recall'):.4f}")

    return best_run.info.run_id


def register_and_promote(run_id: str, model_name: str, client: MlflowClient):
    """
    Registers the model from run_id and promotes it to Production.
    """

    # Step 1 — Register
    model_uri = f"runs:/{run_id}/model"
    print(f"\n Registering model from run {run_id[:8]}...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = result.version
    print(f"  Registered as '{model_name}' version {version}")

    # Step 2 — Add description
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            "Logistic Regression (C=0.1) trained on Wisconsin Breast Cancer dataset. "
            "Best model from 18-run hyperparameter sweep. "
            f"ROC AUC: {client.get_run(run_id).data.metrics['roc_auc']:.4f} | "
            f"Recall: {client.get_run(run_id).data.metrics['recall']:.4f}"
        )
    )

    # Step 3 — Promote to Production
    print(f"\n Promoting version {version} to Production...")
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version,
    )

    print(f"\n{'═'*50}")
    print(f"  Model '{model_name}' v{version} is now in Production")
    print(f"  Load it with:")
    print(f"  mlflow.sklearn.load_model('models:/{model_name}@production')")
    print(f"{'═'*50}\n")

    return version


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    run_id = find_best_run(client, EXPERIMENT_NAME)
    version = register_and_promote(run_id, MODEL_NAME, client)