"""
Step 1: Data Understanding
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/01_data_understanding.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import ID_COLUMN, TARGET_COLUMN  # noqa: E402
from data_loader import load_raw_data  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEP 1: DATA UNDERSTANDING — IBM Telco Customer Churn")

    # ------------------------------------------------------------------
    # 1.1 Load dataset
    # ------------------------------------------------------------------
    df = load_raw_data()

    print_section("1.1 Dataset Loaded Successfully")
    print(f"Source file : {PROJECT_ROOT / 'data/raw/Telco-Customer-Churn.csv'}")
    print(f"Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 1.2 First look
    # ------------------------------------------------------------------
    print_section("1.2 First 5 Rows")
    print(df.head().to_string())

    print_section("1.3 Column Names & Data Types")
    dtype_summary = pd.DataFrame(
        {"Column": df.columns, "Dtype": df.dtypes.astype(str).values, "Non-Null": df.notna().sum().values}
    )
    print(dtype_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 1.4 Feature dictionary
    # ------------------------------------------------------------------
    print_section("1.4 Feature Dictionary (Business Meaning)")

    feature_dictionary = {
        "customerID": "Unique customer identifier (not used for modeling)",
        "gender": "Customer gender (Male / Female)",
        "SeniorCitizen": "Whether customer is 65+ (1 = Yes, 0 = No)",
        "Partner": "Whether customer has a partner (Yes / No)",
        "Dependents": "Whether customer has dependents (Yes / No)",
        "tenure": "Number of months customer has stayed with the company",
        "PhoneService": "Whether customer has phone service (Yes / No)",
        "MultipleLines": "Multiple phone lines (Yes / No / No phone service)",
        "InternetService": "Type of internet (DSL / Fiber optic / No)",
        "OnlineSecurity": "Online security add-on (Yes / No / No internet service)",
        "OnlineBackup": "Online backup add-on",
        "DeviceProtection": "Device protection add-on",
        "TechSupport": "Tech support add-on",
        "StreamingTV": "Streaming TV add-on",
        "StreamingMovies": "Streaming movies add-on",
        "Contract": "Contract type (Month-to-month / One year / Two year)",
        "PaperlessBilling": "Paperless billing enrolled (Yes / No)",
        "PaymentMethod": "How customer pays (Electronic check, Mailed check, etc.)",
        "MonthlyCharges": "Amount charged to customer each month",
        "TotalCharges": "Total amount charged over customer lifetime",
        "Churn": "Target — whether customer left (Yes / No)",
    }

    for col, desc in feature_dictionary.items():
        print(f"  {col:18s} → {desc}")

    # ------------------------------------------------------------------
    # 1.5 Numerical vs categorical split
    # ------------------------------------------------------------------
    print_section("1.5 Feature Types")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features     ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # ------------------------------------------------------------------
    # 1.6 Target variable
    # ------------------------------------------------------------------
    print_section("1.6 Target Variable Analysis — Churn")

    churn_counts = df[TARGET_COLUMN].value_counts()
    churn_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    print(churn_counts.to_string())
    print("\nClass distribution (%):")
    for label, pct in churn_pct.items():
        print(f"  {label}: {pct:.2f}%")

    churn_rate = (df[TARGET_COLUMN] == "Yes").mean() * 100
    print(f"\nOverall churn rate: {churn_rate:.2f}%")
    print("Note: Moderate class imbalance — churners are the minority class (~27%).")

    # ------------------------------------------------------------------
    # 1.7 Missing values
    # ------------------------------------------------------------------
    print_section("1.7 Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No null values detected by pandas isnull().")
    else:
        print(missing.to_string())

    # TotalCharges has whitespace strings that become NaN after numeric conversion
    total_charges_issues = df["TotalCharges"].isna().sum()
    print(f"\nTotalCharges non-numeric / missing after conversion: {total_charges_issues}")
    if total_charges_issues > 0:
        print("These are typically new customers (tenure = 0) — will be handled in cleaning.")

    # ------------------------------------------------------------------
    # 1.8 Duplicates & unique IDs
    # ------------------------------------------------------------------
    print_section("1.8 Duplicate & ID Checks")

    print(f"Duplicate rows          : {df.duplicated().sum()}")
    print(f"Unique customerID count : {df[ID_COLUMN].nunique()} / {len(df)}")
    if df[ID_COLUMN].nunique() == len(df):
        print("All customer IDs are unique — safe to use as identifier only.")

    # ------------------------------------------------------------------
    # 1.9 Numerical summary
    # ------------------------------------------------------------------
    print_section("1.9 Numerical Feature Summary")
    print(df[numeric_cols].describe().round(2).to_string())

    # ------------------------------------------------------------------
    # 1.10 Categorical cardinality
    # ------------------------------------------------------------------
    print_section("1.10 Categorical Cardinality (unique values per column)")

    cat_cardinality = {col: df[col].nunique() for col in categorical_cols if col != ID_COLUMN}
    for col, n in sorted(cat_cardinality.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col:18s}: {n:3d} unique → {df[col].unique()[:5].tolist()} ...")

    # ------------------------------------------------------------------
    # 1.11 Business context
    # ------------------------------------------------------------------
    print_section("1.11 Why Churn Prediction Matters (Business Context)")

    business_points = [
        "Acquiring a new telecom customer costs 5–7× more than retaining an existing one.",
        "A ~27% churn rate means roughly 1 in 4 customers leave — significant revenue leakage.",
        "Predicting churn early lets the company offer targeted retention (discounts, upgrades).",
        "This dataset captures billing, contract, and service usage — key drivers of loyalty.",
    ]
    for i, point in enumerate(business_points, 1):
        print(f"  {i}. {point}")

    print_section("STEP 1 COMPLETE — Ready for Data Cleaning (Step 2)")
    print("Review the outputs above, then confirm to proceed to Step 2.")


if __name__ == "__main__":
    main()
