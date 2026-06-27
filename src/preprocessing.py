"""Data preprocessing utilities for the Telco Churn project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    EXCLUDE_COLUMNS,
    FEATURE_NAMES_FILE,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_TEST_FILE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a Scikit-learn ColumnTransformer for numeric scaling and categorical encoding.

    - Numeric features → StandardScaler (zero mean, unit variance)
    - Categorical features → OneHotEncoder (drop first category to reduce multicollinearity)
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split engineered dataframe into feature matrix X and binary target y."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols].copy()
    y = df["ChurnNumeric"].copy()
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train-test split preserving churn class ratio."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessor on training data and transform both splits."""
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed


def get_processed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names after scaling and one-hot encoding."""
    return list(preprocessor.get_feature_names_out())


def save_preprocessing_artifacts(
    preprocessor: ColumnTransformer,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Save preprocessor, train/test arrays, and feature names to disk."""
    PREPROCESSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.values,
            "y_test": y_test.values,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "test_size_ratio": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        TRAIN_TEST_FILE,
    )

    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return {
        "preprocessor_path": PREPROCESSOR_FILE,
        "train_test_path": TRAIN_TEST_FILE,
        "feature_names_path": FEATURE_NAMES_FILE,
    }


def load_preprocessor() -> ColumnTransformer:
    """Load fitted preprocessor from disk."""
    return joblib.load(PREPROCESSOR_FILE)


def load_train_test_data() -> dict:
    """Load preprocessed train/test arrays and metadata."""
    return joblib.load(TRAIN_TEST_FILE)


def run_preprocessing(df: pd.DataFrame) -> dict:
    """
    Execute the full preprocessing pipeline.

    Steps:
    1. Separate features (X) and target (y)
    2. Stratified train-test split
    3. Fit StandardScaler + OneHotEncoder on train set
    4. Transform both splits
    5. Save artifacts for model training and Flask deployment

    Returns
    -------
    dict
        Summary with shapes, class distribution, and saved file paths.
    """
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    X_train_processed, X_test_processed = fit_transform_preprocessor(
        preprocessor, X_train, X_test
    )
    feature_names = get_processed_feature_names(preprocessor)

    paths = save_preprocessing_artifacts(
        preprocessor=preprocessor,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train,
        X_test_raw=X_test,
        feature_names=feature_names,
    )

    train_churn_rate = y_train.mean() * 100
    test_churn_rate = y_test.mean() * 100

    return {
        "raw_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "processed_features": X_train_processed.shape[1],
        "train_rows": X_train_processed.shape[0],
        "test_rows": X_test_processed.shape[0],
        "train_churn_rate_pct": round(train_churn_rate, 2),
        "test_churn_rate_pct": round(test_churn_rate, 2),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
        "paths": paths,
    }
