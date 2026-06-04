# src/sweep.py

import sys
import os

# Always resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from train import train
import itertools

# ── Hyperparameter Grids ──────────────────────────────────────────────────────

LR_GRID = {
    "model_type": "logistic_regression",
    "params": [
        {"C": c, "max_iter": 1000, "random_state": 42}
        for c in [0.01, 0.1, 1.0, 10.0, 100.0]
    ]
}

RF_GRID = {
    "model_type": "random_forest",
    "params": [
        {"n_estimators": n, "max_depth": d, "random_state": 42}
        for n, d in itertools.product(
            [50, 100, 200],
            [None, 5, 10]
        )
    ]
}

SVM_GRID = {
    "model_type": "svm",
    "params": [
        {"C": c}
        for c in [0.1, 1.0, 10.0, 100.0]
    ]
}

GRIDS = [LR_GRID, RF_GRID, SVM_GRID]

# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    results = []
    total = sum(len(g["params"]) for g in GRIDS)
    count = 0

    print(f"\n{'═'*50}")
    print(f"  Starting sweep — {total} runs total")
    print(f"{'═'*50}\n")

    for grid in GRIDS:
        model_type = grid["model_type"]

        for params in grid["params"]:
            count += 1
            print(f"[{count}/{total}] {model_type} | {params}")

            run_id, metrics = train(model_type=model_type, **params)

            results.append({
                "run_id":     run_id,
                "model_type": model_type,
                "params":     params,
                "metrics":    metrics,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  Sweep complete — {total} runs logged to MLflow")
    print(f"{'═'*50}")

    # Sort by ROC AUC descending
    results.sort(key=lambda x: x["metrics"]["roc_auc"], reverse=True)

    print(f"\n  Top 5 runs by ROC AUC:")
    print(f"  {'Model':<25} {'ROC AUC':>8} {'F1':>8} {'Recall':>8}")
    print(f"  {'─'*55}")
    for r in results[:5]:
        print(
            f"  {r['model_type']:<25}"
            f"  {r['metrics']['roc_auc']:.4f}"
            f"  {r['metrics']['f1_score']:.4f}"
            f"  {r['metrics']['recall']:.4f}"
        )
    print()


if __name__ == "__main__":
    run_sweep()