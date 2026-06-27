"""Data cleaning utilities for the Telco Customer Churn project."""

import pandas as pd

from config import CLEANED_DATA_FILE, ID_COLUMN, TARGET_COLUMN


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all object columns."""
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    return df


def fix_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fix missing TotalCharges values.

    New customers (tenure = 0) have blank TotalCharges in the raw file.
    Business rule: impute with 0 since no charges have accumulated yet.
    """
    df = df.copy()
    missing_mask = df["TotalCharges"].isna()
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        if not (df.loc[missing_mask, "tenure"] == 0).all():
            bad_rows = df.loc[missing_mask & (df["tenure"] != 0)]
            raise ValueError(
                f"TotalCharges missing for {len(bad_rows)} rows with tenure > 0. "
                "Manual review required."
            )
        df.loc[missing_mask, "TotalCharges"] = 0.0

    return df, missing_count


def validate_categorical_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return unexpected categorical values per column."""
    expected = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "MultipleLines": {"Yes", "No", "No phone service"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "OnlineSecurity": {"Yes", "No", "No internet service"},
        "OnlineBackup": {"Yes", "No", "No internet service"},
        "DeviceProtection": {"Yes", "No", "No internet service"},
        "TechSupport": {"Yes", "No", "No internet service"},
        "StreamingTV": {"Yes", "No", "No internet service"},
        "StreamingMovies": {"Yes", "No", "No internet service"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaperlessBilling": {"Yes", "No"},
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        TARGET_COLUMN: {"Yes", "No"},
    }

    issues: dict[str, list[str]] = {}
    for col, allowed in expected.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = sorted(actual - allowed)
        if unexpected:
            issues[col] = unexpected

    return issues


def detect_numeric_outliers(
    df: pd.DataFrame, columns: list[str] | None = None, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """Flag numeric outliers using the IQR method (informational only)."""
    columns = columns or ["tenure", "MonthlyCharges", "TotalCharges"]
    summary_rows = []

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        summary_rows.append(
            {
                "column": col,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(outlier_mask.mean() * 100, 2),
            }
        )

    return pd.DataFrame(summary_rows)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline on raw Telco data."""
    report: dict = {"initial_rows": len(df)}

    df = strip_whitespace(df)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    duplicates_before = int(df.duplicated().sum())
    if duplicates_before > 0:
        df = df.drop_duplicates(keep="first")
    report["duplicates_removed"] = duplicates_before

    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
    if duplicate_ids > 0:
        df = df.drop_duplicates(subset=[ID_COLUMN], keep="first")
    report["duplicate_ids_removed"] = duplicate_ids

    df, missing_total_charges = fix_total_charges(df)
    report["total_charges_imputed"] = missing_total_charges

    cat_issues = validate_categorical_values(df)
    if cat_issues:
        raise ValueError(f"Unexpected categorical values found: {cat_issues}")
    report["categorical_issues"] = cat_issues

    negative_charges = int((df["MonthlyCharges"] < 0).sum() + (df["TotalCharges"] < 0).sum())
    invalid_tenure = int((df["tenure"] < 0).sum())
    if negative_charges > 0 or invalid_tenure > 0:
        raise ValueError("Invalid negative values detected in charges or tenure.")

    inconsistent = df[(df["tenure"] >= 1) & (df["TotalCharges"] < df["MonthlyCharges"])]
    report["total_lt_monthly_count"] = len(inconsistent)

    report["outlier_summary"] = detect_numeric_outliers(df)
    report["final_rows"] = len(df)
    report["columns"] = df.columns.tolist()

    return df, report


def save_cleaned_data(df: pd.DataFrame, filepath=None) -> None:
    """Persist cleaned dataset to CSV."""
    path = filepath or CLEANED_DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
