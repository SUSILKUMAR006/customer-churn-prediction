"""Data loading utilities for the Telco Customer Churn project."""

import pandas as pd

from config import RAW_DATA_FILE


def load_raw_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to data/raw/Telco-Customer-Churn.csv.

    Returns
    -------
    pd.DataFrame
        Raw customer records.
    """
    path = filepath or RAW_DATA_FILE
    df = pd.read_csv(path)

    # TotalCharges is stored as object; some rows contain whitespace-only values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def load_cleaned_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the cleaned dataset from data/processed/."""
    from config import CLEANED_DATA_FILE

    path = filepath or CLEANED_DATA_FILE
    return pd.read_csv(path)


def load_engineered_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the feature-engineered dataset from data/processed/."""
    from config import ENGINEERED_DATA_FILE

    path = filepath or ENGINEERED_DATA_FILE
    return pd.read_csv(path)
