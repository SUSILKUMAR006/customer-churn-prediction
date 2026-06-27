"""Model evaluation utilities for the Telco Churn project."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    BEST_MODEL_FILE,
    LIGHTGBM_MODEL_FILE,
    LOGISTIC_REGRESSION_MODEL_FILE,
    MODEL_REGISTRY_FILE,
    RANDOM_FOREST_MODEL_FILE,
    REPORTS_DIR,
)
from preprocessing import load_train_test_data


def evaluate_model_performance(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Compute standard classification metrics on test data."""
    y_pred = model.predict(X_test)
    
    # Check if the model supports predict_proba (most standard classifiers do)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback to decision_function or binary predictions
        if hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def plot_confusion_matrices(
    evaluated_models: dict[str, tuple[Any, dict[str, float], np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confusion matrices for all models side-by-side."""
    sns.set_theme(style="whitegrid")
    num_models = len(evaluated_models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    
    if num_models == 1:
        axes = [axes]

    for ax, (name, (model, _, y_pred, _)) in zip(axes, evaluated_models.items()):
        cm = confusion_matrix(y_test, y_pred)
        # Format labels clearly
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        # Clean title
        display_name = name.replace("_", " ").title()
        ax.set_title(f"{display_name} Confusion Matrix", fontsize=13, pad=10)
        ax.set_ylabel("Actual Class", fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_evaluation_curves(
    evaluated_models: dict[str, tuple[Any, dict[str, float], np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC and Precision-Recall curves side-by-side for model comparison."""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for consistent plotting
    colors = {"lightgbm": "#1f77b4", "logistic_regression": "#ff7f0e", "random_forest": "#2ca02c"}

    for name, (_, _, _, y_prob) in evaluated_models.items():
        color = colors.get(name, None)
        display_name = name.replace("_", " ").title()

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{display_name} (AUC = {roc_auc:.3f})")

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=color, lw=2, label=f"{display_name} (AUC = {pr_auc:.3f})")

    # Finalize ROC plot
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, pad=10)
    ax1.legend(loc="lower right", frameon=True)

    # Finalize PR plot
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    ax2.plot([0, 1], [no_skill, no_skill], color="gray", lw=1, linestyle="--", label=f"No Skill ({no_skill:.3f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall (PR) Curves", fontsize=13, pad=10)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_evaluation() -> dict[str, Any]:
    """
    Run evaluation on the test set for all trained models.
    
    Steps:
    1. Load preprocessed test arrays
    2. Load trained models
    3. Calculate test metrics for each model
    4. Export confusion matrices and ROC/PR plots
    5. Determine the best model by test F1-score
    6. Update model_registry.json with test metrics
    7. Save the best model to best_model.joblib
    """
    # Load test data
    data = load_train_test_data()
    X_test = data["X_test"]
    y_test = data["y_test"]

    model_paths = {
        "lightgbm": LIGHTGBM_MODEL_FILE,
        "logistic_regression": LOGISTIC_REGRESSION_MODEL_FILE,
        "random_forest": RANDOM_FOREST_MODEL_FILE,
    }

    # Verify models exist
    missing = [k for k, v in model_paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained model files: {missing}. Run Step 6 first.")

    # Load and evaluate each model
    evaluated_models = {}
    test_metrics = {}

    for name, path in model_paths.items():
        model = joblib.load(path)
        metrics = evaluate_model_performance(model, X_test, y_test)
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        evaluated_models[name] = (model, metrics, y_pred, y_prob)
        test_metrics[name] = metrics

    # Plot metrics
    confusion_matrix_path = REPORTS_DIR / "confusion_matrices.png"
    evaluation_curves_path = REPORTS_DIR / "evaluation_curves.png"
    
    plot_confusion_matrices(evaluated_models, y_test, confusion_matrix_path)
    plot_evaluation_curves(evaluated_models, y_test, evaluation_curves_path)

    # Determine best model based on F1-score
    best_model_name = max(test_metrics.keys(), key=lambda k: test_metrics[k]["f1"])
    best_model_metrics = test_metrics[best_model_name]
    best_model_path = model_paths[best_model_name]

    # Save best model
    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, BEST_MODEL_FILE)

    # Update model registry
    registry = {}
    if MODEL_REGISTRY_FILE.exists():
        try:
            with open(MODEL_REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            pass

    # Enrich registry with test evaluation metrics
    registry["primary_model"] = best_model_name
    registry["best_model_path"] = str(BEST_MODEL_FILE)
    registry["evaluation_plots"] = {
        "confusion_matrices": str(confusion_matrix_path),
        "evaluation_curves": str(evaluation_curves_path),
    }

    for name in model_paths.keys():
        if "models" not in registry:
            registry["models"] = {}
        if name not in registry["models"]:
            registry["models"][name] = {}
            
        # Add test set statistics
        registry["models"][name]["test_metrics"] = {
            "accuracy": round(test_metrics[name]["accuracy"], 4),
            "precision": round(test_metrics[name]["precision"], 4),
            "recall": round(test_metrics[name]["recall"], 4),
            "f1": round(test_metrics[name]["f1"], 4),
            "roc_auc": round(test_metrics[name]["roc_auc"], 4),
        }

    with open(MODEL_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)

    return {
        "test_metrics": test_metrics,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics,
        "confusion_matrix_plot": confusion_matrix_path,
        "evaluation_curves_plot": evaluation_curves_path,
        "registry_path": MODEL_REGISTRY_FILE,
    }
