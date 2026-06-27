"""
Step 3: Exploratory Data Analysis (EDA)
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/03_eda.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLEANED_DATA_FILE, REPORTS_DIR, TARGET_COLUMN  # noqa: E402
from data_cleaning import clean_data, save_cleaned_data  # noqa: E402
from data_loader import load_cleaned_data, load_raw_data  # noqa: E402
from eda import churn_rate_table, ensure_churn_numeric, run_eda  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data_for_eda() -> pd.DataFrame:
    """Load cleaned data; generate it if not yet saved."""
    if CLEANED_DATA_FILE.exists():
        return load_cleaned_data()
    df_raw = load_raw_data()
    df_clean, _ = clean_data(df_raw)
    save_cleaned_data(df_clean)
    return df_clean


def main() -> None:
    print_section("STEP 3: EXPLORATORY DATA ANALYSIS — IBM Telco Customer Churn")

    df = load_data_for_eda()
    df = ensure_churn_numeric(df)

    # ------------------------------------------------------------------
    # 3.1 Overall churn overview
    # ------------------------------------------------------------------
    print_section("3.1 Overall Churn Overview")
    overall_rate = df["ChurnNumeric"].mean() * 100
    print(f"Total customers : {len(df):,}")
    print(f"Churned (Yes)     : {(df[TARGET_COLUMN] == 'Yes').sum():,}")
    print(f"Retained (No)     : {(df[TARGET_COLUMN] == 'No').sum():,}")
    print(f"Overall churn rate: {overall_rate:.2f}%")

    # ------------------------------------------------------------------
    # 3.2 Churn by key business drivers
    # ------------------------------------------------------------------
    print_section("3.2 Churn Rate by Contract Type")
    print(churn_rate_table(df, "Contract").to_string(index=False))

    print_section("3.3 Churn Rate by Payment Method")
    print(churn_rate_table(df, "PaymentMethod").to_string(index=False))

    print_section("3.4 Churn Rate by Internet Service")
    print(churn_rate_table(df, "InternetService").to_string(index=False))

    print_section("3.5 Churn Rate by Tenure Bins")
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    df_binned = df.copy()
    df_binned["tenure_bin"] = pd.cut(df_binned["tenure"], bins=bins, labels=labels, include_lowest=True)
    print(churn_rate_table(df_binned, "tenure_bin").to_string(index=False))

    # ------------------------------------------------------------------
    # 3.6 Numeric summary by churn
    # ------------------------------------------------------------------
    print_section("3.6 Numeric Feature Summary by Churn")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary = df.groupby(TARGET_COLUMN)[numeric_cols].agg(["mean", "median", "std"]).round(2)
    print(summary.to_string())

    # ------------------------------------------------------------------
    # 3.7 Correlation insights
    # ------------------------------------------------------------------
    print_section("3.7 Key Correlation Insights (Numeric)")
    corr_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "ChurnNumeric"]
    corr = df[corr_cols].corr()["ChurnNumeric"].drop("ChurnNumeric").sort_values(key=abs, ascending=False)
    for feat, val in corr.items():
        direction = "↑ churn risk" if val > 0 else "↓ churn risk"
        print(f"  {feat:16s}: {val:+.3f}  ({direction})")

    # ------------------------------------------------------------------
    # 3.8 Top churn drivers (combined summary)
    # ------------------------------------------------------------------
    print_section("3.8 Top 10 Highest-Churn Segments")
    results = run_eda(df, REPORTS_DIR)
    top_segments = results["summary"].head(10)
    print(top_segments.to_string(index=False))

    # ------------------------------------------------------------------
    # 3.9 Key EDA findings (business language)
    # ------------------------------------------------------------------
    print_section("3.9 Key EDA Findings")
    findings = [
        "Month-to-month contracts have the highest churn — lack of commitment is a major driver.",
        "Electronic check payers churn more than auto-pay (credit card / bank transfer) customers.",
        "Fiber optic customers churn more than DSL — possibly price sensitivity on premium plans.",
        "Customers in first 6 months (tenure 0-6) show the highest churn — critical onboarding window.",
        "Higher MonthlyCharges correlates with higher churn — price/value perception matters.",
        "Longer tenure strongly correlates with lower churn — loyalty builds over time.",
    ]
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    # ------------------------------------------------------------------
    # 3.10 Saved artifacts
    # ------------------------------------------------------------------
    print_section("3.10 Saved Figures & Reports")
    print(f"Output directory: {REPORTS_DIR}")
    for fig_path in results["figures"]:
        print(f"  ✓ {fig_path.name}")
    print(f"  ✓ {results['summary_path'].name}")

    print_section("STEP 3 COMPLETE — Ready for Feature Engineering (Step 4)")
    print("Review the plots in reports/figures/, then confirm to proceed to Step 4.")


if __name__ == "__main__":
    main()
