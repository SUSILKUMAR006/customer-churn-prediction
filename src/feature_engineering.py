"""Feature engineering utilities for the Telco Churn project."""

from __future__ import annotations

import pandas as pd

from config import ENGINEERED_DATA_FILE, SERVICE_COLUMNS, TARGET_COLUMN


# Columns created by engineer_features (excluding ChurnNumeric helper)
ENGINEERED_FEATURE_COLUMNS = [
    "AvgMonthlySpend",
    "TenureGroup",
    "ServiceCount",
    "HasInternet",
    "HasPhone",
    "IsMonthToMonth",
    "IsAutoPay",
    "IsElectronicCheck",
    "IsFiber",
    "HasFamily",
    "ChargeIncreaseRatio",
    "IsNewCustomer",
    "HasStreaming",
    "HasSupportBundle",
]


def _service_is_active(series: pd.Series) -> pd.Series:
    """Return 1 when a service add-on is actively subscribed."""
    return series.eq("Yes").astype(int)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Create domain-driven features from cleaned Telco data.

    Features are designed from EDA insights:
    - Contract type and payment method strongly drive churn
    - Tenure and charge patterns differentiate churners
    - Service adoption indicates engagement level

    Returns
    -------
    df_engineered : pd.DataFrame
        Original columns plus engineered features.
    report : dict
        Summary of features created.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Billing & tenure features
    # ------------------------------------------------------------------

    # Average monthly spend over customer lifetime (avoid div-by-zero)
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # Current bill vs historical average — values > 1 suggest recent price increases
    df["ChargeIncreaseRatio"] = df["MonthlyCharges"] / df["AvgMonthlySpend"].replace(0, pd.NA)
    df["ChargeIncreaseRatio"] = df["ChargeIncreaseRatio"].fillna(1.0)

    # Tenure segments from EDA (0-6 months = highest churn window)
    tenure_bins = [-1, 6, 12, 24, 48, 72]
    tenure_labels = ["New (0-6)", "Early (7-12)", "Mid (13-24)", "Loyal (25-48)", "Veteran (49-72)"]
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
    ).astype(str)

    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    # ------------------------------------------------------------------
    # 2. Service engagement features
    # ------------------------------------------------------------------

    df["ServiceCount"] = sum(_service_is_active(df[col]) for col in SERVICE_COLUMNS)

    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    df["HasPhone"] = (df["PhoneService"] == "Yes").astype(int)
    df["IsFiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    df["HasSupportBundle"] = (
        (df["OnlineSecurity"] == "Yes")
        | (df["TechSupport"] == "Yes")
        | (df["DeviceProtection"] == "Yes")
    ).astype(int)

    # ------------------------------------------------------------------
    # 3. Contract & payment risk flags (top EDA churn drivers)
    # ------------------------------------------------------------------

    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    df["IsAutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # ------------------------------------------------------------------
    # 4. Demographic composite
    # ------------------------------------------------------------------

    df["HasFamily"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

    # Binary target for modeling convenience (preprocessing may use this)
    df["ChurnNumeric"] = (df[TARGET_COLUMN] == "Yes").astype(int)

    report = {
        "rows": len(df),
        "new_features": ENGINEERED_FEATURE_COLUMNS + ["ChurnNumeric"],
        "feature_definitions": {
            "AvgMonthlySpend": "TotalCharges / max(tenure, 1) — lifetime average monthly spend",
            "TenureGroup": "Binned tenure: New/Early/Mid/Loyal/Veteran",
            "ServiceCount": "Count of subscribed add-on services (0-6)",
            "HasInternet": "1 if InternetService is DSL or Fiber",
            "HasPhone": "1 if PhoneService is Yes",
            "IsMonthToMonth": "1 if Contract is Month-to-month (high churn segment)",
            "IsAutoPay": "1 if payment is automatic (bank/card)",
            "IsElectronicCheck": "1 if payment is Electronic check (high churn segment)",
            "IsFiber": "1 if InternetService is Fiber optic",
            "HasFamily": "1 if customer has Partner or Dependents",
            "ChargeIncreaseRatio": "MonthlyCharges / AvgMonthlySpend — bill vs history",
            "IsNewCustomer": "1 if tenure <= 6 months (critical onboarding window)",
            "HasStreaming": "1 if StreamingTV or StreamingMovies is Yes",
            "HasSupportBundle": "1 if any security/support/protection add-on is Yes",
            "ChurnNumeric": "Target encoded as 1=Yes, 0=No",
        },
    }

    return df, report


def get_feature_churn_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean engineered feature values for churned vs retained customers."""
    numeric_engineered = [
        c
        for c in ENGINEERED_FEATURE_COLUMNS
        if c != "TenureGroup" and c in df.columns
    ]
    comparison = (
        df.groupby(TARGET_COLUMN)[numeric_engineered]
        .mean()
        .T
        .rename(columns={"No": "retained_mean", "Yes": "churned_mean"})
    )
    comparison["difference"] = comparison["churned_mean"] - comparison["retained_mean"]
    return comparison.round(3)


def save_engineered_data(df: pd.DataFrame, filepath=None) -> None:
    """Persist engineered dataset to CSV."""
    path = filepath or ENGINEERED_DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
