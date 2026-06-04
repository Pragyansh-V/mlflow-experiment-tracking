# MLflow Experiment Tracking — Breast Cancer Classification

A complete MLOps experiment tracking pipeline built on the Wisconsin Breast Cancer dataset, demonstrating the full lifecycle from raw data to a registered production model.

> **"Reproducibility is the difference between a prototype and a production system."**

---

## Project Overview

Most ML engineers hit a wall early: *"I trained 30 models last week. I think run 17 was the best but I'm not sure what parameters I used."* This project closes that gap by making experiment tracking a first-class citizen of the ML workflow.

**What this project demonstrates:**
- Loading and preprocessing a real dataset from HuggingFace Hub
- Tracking every experiment — params, metrics, artifacts — with MLflow
- Running a structured hyperparameter sweep across 3 model types (18 runs)
- Registering the best model to the MLflow Model Registry with a `@production` alias
- Loading the production model by alias and running final evaluation

---

## Results

| Model | C | ROC AUC | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.1 | **0.9977** | **0.9756** | **0.9524** |
| Logistic Regression | 0.01 | 0.9977 | 0.9231 | 0.8571 |
| Random Forest | 100, depth=5 | 0.9950 | 0.9630 | 0.9286 |
| SVM | 1.0 | 0.9947 | 0.9630 | 0.9286 |

**Best model:** Logistic Regression (`C=0.1`) — selected for best ROC AUC and recall simultaneously.

### Production Model — Final Evaluation

```
Accuracy  : 98.25%
Precision : 100.00%   ← zero false positives
Recall    : 95.24%    ← catches 40/42 malignant cases
F1 Score  : 97.56%
ROC AUC   : 99.77%
```

---

## The MLOps Lifecycle

```
HuggingFace Dataset
      ↓
data_loader.py    — loads, cleans, splits Wisconsin Breast Cancer data
      ↓
train.py          — trains LR / RF / SVM with full MLflow tracking
      ↓
sweep.py          — 18-run hyperparameter sweep across all 3 models
      ↓
register_model.py — finds best run by ROC AUC, registers to Model Registry
      ↓
evaluate.py       — loads @production model, runs final evaluation, logs artifacts
```

---

## Project Structure

```
mlflow-experiment-tracking/
├── src/
│   ├── data_loader.py      # HuggingFace → pandas pipeline
│   ├── train.py            # Training loop with MLflow tracking
│   ├── sweep.py            # Hyperparameter sweep (18 runs)
│   ├── register_model.py   # Model Registry — promote to @production
│   └── evaluate.py         # Load @production model, evaluate, log artifacts
├── artifacts/
│   ├── confusion_matrix_*.png
│   ├── eval_confusion_matrix_*.png
│   └── roc_curve_*.png
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone
git clone https://github.com/Pragyansh-V/mlflow-experiment-tracking.git
cd mlflow-experiment-tracking

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train a single model
```bash
python src/train.py --model_type logistic_regression --C 0.1
python src/train.py --model_type random_forest --n_estimators 100 --max_depth 5
python src/train.py --model_type svm --C 1.0
```

### 2. Run the full hyperparameter sweep
```bash
python src/sweep.py
```

### 3. Launch MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

### 4. Register the best model
```bash
python src/register_model.py
```

### 5. Evaluate the production model
```bash
python src/evaluate.py
```

---

## Key Insights

**Why Logistic Regression wins:** Lower regularisation (`C=0.1`) produces better probability calibration — reflected in the superior ROC AUC. This is non-obvious; you'd expect a trade-off between precision and recall, but the right regularisation strength eliminates it on this dataset.

**Why ROC AUC over accuracy:** In a clinical context, a model that correctly discriminates malignant from benign cases across all decision thresholds is more valuable than one that maximises a single threshold's accuracy. ROC AUC measures exactly that.

**Why recall matters more than precision here:** A false negative (telling a cancer patient they're healthy) is clinically more dangerous than a false positive (unnecessary follow-up tests). The production model achieves 95.24% recall with perfect precision — the best possible outcome on this dataset.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| MLflow 3.x | Experiment tracking, Model Registry |
| scikit-learn | LR, RF, SVM classifiers |
| HuggingFace Datasets | Data loading |
| pandas | Data manipulation |
| matplotlib / seaborn | Artifact visualisation |
| SQLite | MLflow backend store |

---

## Future Extensions

- [ ] Retrofit MLflow autolog onto the [Emotion Classifier (Project 2)](https://github.com/Pragyansh-V)
- [ ] Migrate tracking backend to Dagshub for remote experiment storage
- [ ] Add Optuna integration for Bayesian hyperparameter optimisation
- [ ] Add threshold tuning to push recall above 98% for clinical deployment

---

## Part of the HuggingFace Portfolio Series

This is **Project 7** in a series of ML/AI portfolio projects:

| # | Project | Key Skills |
|---|---|---|
| 1 | Sentiment Audit | HuggingFace pipelines |
| 2 | Emotion Classifier | LoRA fine-tuning |
| 3 | RAG Pipeline | Retrieval-Augmented Generation |
| 4 | Bias Audit | Fairness evaluation |
| 5 | LangGraph Agent | Agentic workflows |
| 6 | Vertex AI Deployment | Cloud MLOps |
| **7** | **MLflow Tracking** | **Experiment tracking, Model Registry** |