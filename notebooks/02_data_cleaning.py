"""
Step 2: Data Cleaning
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/02_data_cleaning.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLEANED_DATA_FILE, TARGET_COLUMN  # noqa: E402
from data_cleaning import clean_data, save_cleaned_data  # noqa: E402
from data_loader import load_raw_data  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEP 2: DATA CLEANING — IBM Telco Customer Churn")

    # ------------------------------------------------------------------
    # 2.1 Load raw data
    # ------------------------------------------------------------------
    df_raw = load_raw_data()
    print_section("2.1 Starting Point")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Missing TotalCharges (before cleaning): {df_raw['TotalCharges'].isna().sum()}")

    # ------------------------------------------------------------------
    # 2.2 Apply cleaning pipeline
    # ------------------------------------------------------------------
    print_section("2.2 Cleaning Steps Applied")
    cleaning_steps = [
        "Strip whitespace from all text columns",
        "Enforce correct data types (int/float)",
        "Remove duplicate rows (if any)",
        "Remove duplicate customer IDs (if any)",
        "Impute missing TotalCharges with 0 for new customers (tenure = 0)",
        "Validate all categorical values against expected levels",
        "Check logical consistency (no negative charges/tenure)",
        "Flag numeric outliers using IQR (informational — not removed)",
    ]
    for i, step in enumerate(cleaning_steps, 1):
        print(f"  {i}. {step}")

    df_clean, report = clean_data(df_raw)

    # ------------------------------------------------------------------
    # 2.3 Cleaning report
    # ------------------------------------------------------------------
    print_section("2.3 Cleaning Report")
    print(f"Initial rows              : {report['initial_rows']:,}")
    print(f"Duplicates removed        : {report['duplicates_removed']}")
    print(f"Duplicate IDs removed     : {report['duplicate_ids_removed']}")
    print(f"TotalCharges imputed (0)  : {report['total_charges_imputed']}")
    print(f"Rows where Total < Monthly: {report['total_lt_monthly_count']} (edge cases, kept)")
    print(f"Final rows                : {report['final_rows']:,}")
    print(f"Missing values (final)    : {df_clean.isnull().sum().sum()}")

    print("\nOutlier summary (IQR method — flagged, not removed):")
    print(report["outlier_summary"].to_string(index=False))

    # ------------------------------------------------------------------
    # 2.4 Before vs after: TotalCharges fix
    # ------------------------------------------------------------------
    print_section("2.4 TotalCharges Fix — New Customers (tenure = 0)")
    new_customers = df_clean[df_clean["tenure"] == 0][
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", TARGET_COLUMN]
    ]
    print(f"Customers with tenure = 0: {len(new_customers)}")
    print(new_customers.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 2.5 Data type verification
    # ------------------------------------------------------------------
    print_section("2.5 Final Data Types")
    dtype_df = pd.DataFrame(
        {"Column": df_clean.columns, "Dtype": df_clean.dtypes.astype(str).values}
    )
    print(dtype_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 2.6 Save cleaned dataset
    # ------------------------------------------------------------------
    save_cleaned_data(df_clean)
    print_section("2.6 Cleaned Dataset Saved")
    print(f"Output file: {CLEANED_DATA_FILE}")
    print(f"Shape      : {df_clean.shape}")

    print_section("STEP 2 COMPLETE — Ready for EDA (Step 3)")
    print("Review the outputs above, then confirm to proceed to Step 3.")


if __name__ == "__main__":
    main()
