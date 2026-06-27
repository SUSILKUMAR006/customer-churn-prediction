"""Feature importance and model explainability utilities for the Telco Churn project."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import BEST_MODEL_FILE, FEATURE_NAMES_FILE, REPORTS_DIR


def get_feature_importances(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importances or coefficients from a trained model.
    
    Parameters
    ----------
    model : ClassifierMixin
        The trained classifier.
    feature_names : list[str]
        List of feature names matching the model inputs.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'feature' and 'importance', sorted by importance descending.
    """
    # 1. Tree-based models (LightGBM, Random Forest)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    
    # 2. Linear models (Logistic Regression)
    elif hasattr(model, "coef_"):
        # For standardized features, absolute coefficient represents relative importance
        importances = np.abs(model.coef_[0])
    
    else:
        raise ValueError(f"Model type {type(model)} does not support feature importances or coefficients.")

    # Validate length
    if len(importances) != len(feature_names):
        # Fallback if names mismatch
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    
    # Clean feature names for readability (e.g. remove "num__" or "cat__")
    df["feature_clean"] = df["feature"].apply(
        lambda x: x.replace("num__", "").replace("cat__", "").replace("_", " ").title()
    )
    
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def plot_feature_importances(df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """Plot horizontal bar chart of top N feature importances."""
    sns.set_theme(style="whitegrid")
    
    # Select top N features
    plot_df = df.head(top_n).copy()
    
    # Normalize importance to sum to 1 (or percentage) for tree models to make comparison intuitive
    if plot_df["importance"].sum() > 0:
        plot_df["importance_pct"] = (plot_df["importance"] / df["importance"].sum()) * 100
    else:
        plot_df["importance_pct"] = plot_df["importance"]

    plt.figure(figsize=(10, 6.5))
    
    # Create horizontal bar plot
    ax = sns.barplot(
        x="importance_pct",
        y="feature_clean",
        data=plot_df,
        palette="Blues_r",
        hue="feature_clean",
        legend=False,
    )
    
    # Customize titles and labels
    plt.title(f"Top {top_n} Most Influential Features for Customer Churn", fontsize=14, pad=15)
    plt.xlabel("Relative Importance (%)", fontsize=11, labelpad=10)
    plt.ylabel("", fontsize=11)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(
            width + 0.2,
            p.get_y() + p.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
            weight="bold"
        )

    plt.xlim(0, plot_df["importance_pct"].max() * 1.15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_explainability() -> dict[str, Any]:
    """
    Run explainability pipeline.
    
    Steps:
    1. Load best model
    2. Load feature names
    3. Extract importances
    4. Save horizontal bar plot
    """
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing best model file: {BEST_MODEL_FILE}. Run Step 7 first.")
        
    if not FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(f"Missing feature names file: {FEATURE_NAMES_FILE}. Run Step 5 first.")

    # Load artifacts
    model = joblib.load(BEST_MODEL_FILE)
    with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # Calculate importances
    importances_df = get_feature_importances(model, feature_names)

    # Plot
    output_plot_path = REPORTS_DIR / "feature_importance.png"
    plot_feature_importances(importances_df, output_plot_path)

    # Save importance CSV for downstream analytics
    output_csv_path = REPORTS_DIR / "feature_importance_summary.csv"
    importances_df.to_csv(output_csv_path, index=False)

    return {
        "importances_df": importances_df,
        "feature_importance_plot": output_plot_path,
        "feature_importance_csv": output_csv_path,
        "model_type": type(model).__name__,
    }
