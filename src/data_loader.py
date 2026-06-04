# src/data_loader.py

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(test_size=0.2, random_state=42):
    """
    Loads the Wisconsin Breast Cancer dataset from HuggingFace,
    converts to pandas, cleans it, and returns train/test splits.

    Target: 'diagnosis' — M (malignant) = 1, B (benign) = 0

    Returns:
        X_train, X_test, y_train, y_test (all pandas DataFrames/Series)
    """

    # Load from HuggingFace Hub
    dataset = load_dataset("scikit-learn/breast-cancer-wisconsin", split="train")

    # Convert to pandas
    df = dataset.to_pandas()

    # Drop junk columns — id has no predictive value, Unnamed: 32 is empty
    df = df.drop(columns=["id", "Unnamed: 32"])

    # Encode target: M (malignant) = 1, B (benign) = 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nTarget distribution:\n{df['diagnosis'].value_counts()}")
    print(f"\nFeature columns ({len(df.columns) - 1}): {list(df.drop(columns=['diagnosis']).columns)}")

    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Train/test split — stratify preserves class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_breast_cancer_data()