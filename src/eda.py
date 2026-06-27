"""Exploratory Data Analysis utilities for the Telco Churn project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import REPORTS_DIR, TARGET_COLUMN

# Consistent plot styling across all figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def ensure_churn_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary ChurnNumeric column (1 = Yes, 0 = No) for analysis."""
    df = df.copy()
    df["ChurnNumeric"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    return df


def churn_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute churn count, total, and churn rate per category."""
    summary = (
        df.groupby(column)[TARGET_COLUMN]
        .agg(churn_count=lambda s: (s == "Yes").sum(), total="count")
        .reset_index()
    )
    summary["churn_rate"] = (summary["churn_count"] / summary["total"] * 100).round(2)
    return summary.sort_values("churn_rate", ascending=False)


def plot_churn_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar + pie chart of overall churn class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#2ecc71", "#e74c3c"]

    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=axes[0], palette=colors, legend=False)
    axes[0].set_title("Churn Class Counts")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Customers")

    axes[1].pie(
        counts.values,
        labels=[f"{label}\n({val:,})" for label, val in counts.items()],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[1].set_title("Churn Class Proportion")

    path = output_dir / "01_churn_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_churn_by_category(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
    order: list | None = None,
) -> Path:
    """Bar chart of churn rate (%) by a categorical feature."""
    summary = churn_rate_table(df, column)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_order = order or summary[column].tolist()
    sns.barplot(
        data=summary,
        x=column,
        y="churn_rate",
        order=plot_order,
        hue=column,
        palette="RdYlGn_r",
        ax=ax,
        legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Churn Rate (%)")
    ax.axhline(df["ChurnNumeric"].mean() * 100, color="gray", linestyle="--", label="Overall avg")
    ax.legend()

    for i, row in summary.iterrows():
        idx = plot_order.index(row[column]) if row[column] in plot_order else i
        ax.text(idx, row["churn_rate"] + 0.8, f"{row['churn_rate']:.1f}%", ha="center", fontsize=9)

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tenure_analysis(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Tenure distribution and binned churn rate."""
    # Histogram by churn
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
        subset = df[df[TARGET_COLUMN] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.6, label=f"Churn={label}", color=color)
    ax.set_title("Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend()
    hist_path = output_dir / "05_tenure_distribution.png"
    fig.savefig(hist_path)
    plt.close(fig)

    # Binned churn rate
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    df_binned = df.copy()
    df_binned["tenure_bin"] = pd.cut(df_binned["tenure"], bins=bins, labels=labels, include_lowest=True)
    bin_path = plot_churn_by_category(
        df_binned,
        "tenure_bin",
        "Churn Rate by Tenure Group (months)",
        output_dir / "06_churn_by_tenure_bins.png",
        order=labels,
    )
    return hist_path, bin_path


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Box plots of numeric features split by churn."""
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=col, hue=TARGET_COLUMN, ax=ax, palette=["#2ecc71", "#e74c3c"], legend=False)
        ax.set_title(f"{col} by Churn")
        ax.set_xlabel("Churn")

    path = output_dir / "07_numeric_boxplots_by_churn.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Correlation heatmap for numeric features including binary churn."""
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "ChurnNumeric"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")
    path = output_dir / "08_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_churn_by_demographics(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart for demographic features."""
    demo_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, demo_cols):
        if col == "SeniorCitizen":
            plot_df = df.copy()
            plot_df[col] = plot_df[col].map({0: "No", 1: "Yes"})
        else:
            plot_df = df

        summary = churn_rate_table(plot_df, col)
        sns.barplot(data=summary, x=col, y="churn_rate", hue=col, ax=ax, palette="RdYlGn_r", legend=False)
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate (%)")
        ax.axhline(df["ChurnNumeric"].mean() * 100, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("Churn Rate by Demographics", fontsize=14, y=1.02)
    path = output_dir / "09_churn_by_demographics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_monthly_charges_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """KDE plot of MonthlyCharges by churn."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue=TARGET_COLUMN, fill=True, alpha=0.4, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.set_xlabel("Monthly Charges ($)")
    path = output_dir / "10_monthly_charges_kde.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Key churn rates for top business drivers."""
    key_features = [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "PaperlessBilling",
        "Partner",
        "Dependents",
    ]
    rows = []
    for col in key_features:
        table = churn_rate_table(df, col)
        for _, row in table.iterrows():
            rows.append(
                {
                    "feature": col,
                    "category": row[col],
                    "total": row["total"],
                    "churn_count": row["churn_count"],
                    "churn_rate_pct": row["churn_rate"],
                }
            )
    return pd.DataFrame(rows).sort_values("churn_rate_pct", ascending=False)


def run_eda(df: pd.DataFrame, output_dir: Path | None = None) -> dict:
    """
    Run full EDA pipeline and save all figures.

    Returns dict with paths to saved figures and summary tables.
    """
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_churn_numeric(df)
    results: dict = {"figures": [], "summary": None}

    results["figures"].append(plot_churn_distribution(df, output_dir))
    results["figures"].append(
        plot_churn_by_category(
            df,
            "Contract",
            "Churn Rate by Contract Type",
            output_dir / "02_churn_by_contract.png",
            order=["Month-to-month", "One year", "Two year"],
        )
    )
    results["figures"].append(
        plot_churn_by_category(
            df,
            "PaymentMethod",
            "Churn Rate by Payment Method",
            output_dir / "03_churn_by_payment_method.png",
        )
    )
    results["figures"].append(
        plot_churn_by_category(
            df,
            "InternetService",
            "Churn Rate by Internet Service",
            output_dir / "04_churn_by_internet_service.png",
            order=["No", "DSL", "Fiber optic"],
        )
    )
    results["figures"].extend(plot_tenure_analysis(df, output_dir))
    results["figures"].append(plot_numeric_distributions(df, output_dir))
    results["figures"].append(plot_correlation_heatmap(df, output_dir))
    results["figures"].append(plot_churn_by_demographics(df, output_dir))
    results["figures"].append(plot_monthly_charges_distribution(df, output_dir))

    summary = generate_eda_summary(df)
    summary_path = output_dir / "eda_churn_summary.csv"
    summary.to_csv(summary_path, index=False)
    results["summary"] = summary
    results["summary_path"] = summary_path

    return results
