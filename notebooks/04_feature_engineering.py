"""
Step 4: Feature Engineering
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/04_feature_engineering.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLEANED_DATA_FILE, ENGINEERED_DATA_FILE, TARGET_COLUMN  # noqa: E402
from data_cleaning import clean_data, save_cleaned_data  # noqa: E402
from data_loader import load_cleaned_data, load_raw_data  # noqa: E402
from feature_engineering import (  # noqa: E402
    ENGINEERED_FEATURE_COLUMNS,
    engineer_features,
    get_feature_churn_comparison,
    save_engineered_data,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data_for_engineering() -> pd.DataFrame:
    """Load cleaned data; generate it if not yet saved."""
    if CLEANED_DATA_FILE.exists():
        return load_cleaned_data()
    df_raw = load_raw_data()
    df_clean, _ = clean_data(df_raw)
    save_cleaned_data(df_clean)
    return df_clean


def main() -> None:
    print_section("STEP 4: FEATURE ENGINEERING — IBM Telco Customer Churn")

    df = load_data_for_engineering()
    print_section("4.1 Starting Point")
    print(f"Cleaned data shape: {df.shape}")

    # ------------------------------------------------------------------
    # 4.2 Apply feature engineering
    # ------------------------------------------------------------------
    df_eng, report = engineer_features(df)

    print_section("4.2 New Features Created")
    for i, feat in enumerate(ENGINEERED_FEATURE_COLUMNS, 1):
        definition = report["feature_definitions"][feat]
        print(f"  {i:2d}. {feat:22s} → {definition}")

    # ------------------------------------------------------------------
    # 4.3 Sample engineered records
    # ------------------------------------------------------------------
    print_section("4.3 Sample Engineered Records")
    sample_cols = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        *ENGINEERED_FEATURE_COLUMNS[:6],
        TARGET_COLUMN,
    ]
    print(df_eng[sample_cols].head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 4.4 Feature statistics
    # ------------------------------------------------------------------
    print_section("4.4 Engineered Feature Statistics")
    numeric_feats = [c for c in ENGINEERED_FEATURE_COLUMNS if c != "TenureGroup"]
    print(df_eng[numeric_feats].describe().round(3).to_string())

    print("\nTenureGroup distribution:")
    print(df_eng["TenureGroup"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 4.5 Churned vs retained comparison
    # ------------------------------------------------------------------
    print_section("4.5 Feature Means: Churned vs Retained")
    comparison = get_feature_churn_comparison(df_eng)
    print(comparison.to_string())

    print("\nKey signals (largest absolute differences):")
    top_diff = comparison.copy()
    top_diff["abs_diff"] = top_diff["difference"].abs()
    top_diff = top_diff.sort_values("abs_diff", ascending=False).head(5)
    for feat, row in top_diff.iterrows():
        print(
            f"  {feat:22s}: retained={row['retained_mean']:.3f}, "
            f"churned={row['churned_mean']:.3f}, diff={row['difference']:+.3f}"
        )

    # ------------------------------------------------------------------
    # 4.6 TenureGroup churn rates (validate EDA insight)
    # ------------------------------------------------------------------
    print_section("4.6 TenureGroup Churn Validation")
    tenure_churn = (
        df_eng.groupby("TenureGroup")[TARGET_COLUMN]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )
    print(tenure_churn.to_string())

    # ------------------------------------------------------------------
    # 4.7 Save engineered dataset
    # ------------------------------------------------------------------
    save_engineered_data(df_eng)
    print_section("4.7 Engineered Dataset Saved")
    print(f"Output file : {ENGINEERED_DATA_FILE}")
    print(f"Final shape : {df_eng.shape} ({len(ENGINEERED_FEATURE_COLUMNS)} new features + ChurnNumeric)")

    print_section("STEP 4 COMPLETE — Ready for Data Preprocessing (Step 5)")
    print("Review the features above, then confirm to proceed to Step 5.")


if __name__ == "__main__":
    main()
