"""Model training utilities for the Telco Churn project."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    BEST_MODEL_FILE,
    LIGHTGBM_MODEL_FILE,
    LOGISTIC_REGRESSION_MODEL_FILE,
    MODEL_REGISTRY_FILE,
    PRIMARY_MODEL_NAME,
    RANDOM_FOREST_MODEL_FILE,
    RANDOM_STATE,
    TRAIN_TEST_FILE,
)
from preprocessing import load_train_test_data, run_preprocessing


def build_models() -> dict[str, ClassifierMixin]:
    """Define model candidates with imbalance-aware settings."""
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_model(model: ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    """Fit a single classifier on training data."""
    model.fit(X_train, y_train)
    return model


def save_model(model: ClassifierMixin, filepath) -> None:
    """Persist a trained model to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def ensure_training_data() -> dict:
    """Load train/test arrays; run preprocessing if artifacts are missing."""
    if not TRAIN_TEST_FILE.exists():
        from data_loader import load_engineered_data
        from feature_engineering import engineer_features, save_engineered_data
        from data_cleaning import clean_data, save_cleaned_data
        from data_loader import load_raw_data
        from config import ENGINEERED_DATA_FILE

        if ENGINEERED_DATA_FILE.exists():
            df = load_engineered_data()
        else:
            df_raw = load_raw_data()
            df_clean, _ = clean_data(df_raw)
            save_cleaned_data(df_clean)
            df, _ = engineer_features(df_clean)
            save_engineered_data(df)

        run_preprocessing(df)

    return load_train_test_data()


def run_training() -> dict[str, Any]:
    """
    Train LightGBM, Logistic Regression, and Random Forest.

    Returns
    -------
    dict
        Training summary with model paths and basic train-set metrics.
    """
    data = ensure_training_data()
    X_train = data["X_train"]
    y_train = data["y_train"]

    model_paths = {
        "lightgbm": LIGHTGBM_MODEL_FILE,
        "logistic_regression": LOGISTIC_REGRESSION_MODEL_FILE,
        "random_forest": RANDOM_FOREST_MODEL_FILE,
    }

    models = build_models()
    registry: dict[str, Any] = {"models": {}, "primary_model": PRIMARY_MODEL_NAME}

    for name, model in models.items():
        trained = train_model(model, X_train, y_train)
        save_model(trained, model_paths[name])

        train_pred = trained.predict(X_train)
        train_accuracy = float((train_pred == y_train).mean())

        registry["models"][name] = {
            "path": str(model_paths[name]),
            "train_accuracy": round(train_accuracy, 4),
            "train_churn_rate_actual": round(float(y_train.mean()), 4),
            "train_churn_rate_predicted": round(float(train_pred.mean()), 4),
            "hyperparameters": trained.get_params(),
        }

    # Primary model (LightGBM) also saved as best_model placeholder until Step 7 evaluation
    save_model(joblib.load(LIGHTGBM_MODEL_FILE), BEST_MODEL_FILE)
    registry["best_model_path"] = str(BEST_MODEL_FILE)
    registry["note"] = (
        "best_model.joblib currently mirrors LightGBM; "
        "Step 7 evaluation may update based on test F1-score."
    )

    MODEL_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)

    return {
        "models_trained": list(models.keys()),
        "train_rows": int(len(y_train)),
        "feature_count": int(X_train.shape[1]),
        "registry_path": MODEL_REGISTRY_FILE,
        "model_paths": {k: str(v) for k, v in model_paths.items()},
        "registry": registry,
    }


def load_model(name: str = PRIMARY_MODEL_NAME) -> ClassifierMixin:
    """Load a trained model by registry name."""
    path_map = {
        "lightgbm": LIGHTGBM_MODEL_FILE,
        "logistic_regression": LOGISTIC_REGRESSION_MODEL_FILE,
        "random_forest": RANDOM_FOREST_MODEL_FILE,
        "best": BEST_MODEL_FILE,
    }
    if name not in path_map:
        raise ValueError(f"Unknown model: {name}. Choose from {list(path_map)}")
    return joblib.load(path_map[name])
