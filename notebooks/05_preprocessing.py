"""
Step 5: Data Preprocessing
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/05_preprocessing.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    ENGINEERED_DATA_FILE,
    NUMERIC_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)
from data_cleaning import clean_data, save_cleaned_data  # noqa: E402
from data_loader import load_cleaned_data, load_engineered_data, load_raw_data  # noqa: E402
from feature_engineering import engineer_features, save_engineered_data  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data_for_preprocessing():
    """Load engineered data; run prior pipeline steps if needed."""
    if ENGINEERED_DATA_FILE.exists():
        return load_engineered_data()

    if Path(PROJECT_ROOT / "data" / "processed" / "cleaned_telco_churn.csv").exists():
        from config import CLEANED_DATA_FILE

        df = load_cleaned_data()
    else:
        df_raw = load_raw_data()
        df, _ = clean_data(df_raw)
        save_cleaned_data(df)

    df_eng, _ = engineer_features(df)
    save_engineered_data(df_eng)
    return df_eng


def main() -> None:
    print_section("STEP 5: DATA PREPROCESSING — IBM Telco Customer Churn")

    df = load_data_for_preprocessing()
    print_section("5.1 Input Data")
    print(f"Engineered data shape: {df.shape}")

    # ------------------------------------------------------------------
    # 5.2 Preprocessing strategy
    # ------------------------------------------------------------------
    print_section("5.2 Preprocessing Strategy")
    steps = [
        f"Select {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features",
        f"Stratified train-test split ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, random_state={RANDOM_STATE})",
        "Numeric features → StandardScaler (fit on train only)",
        "Categorical features → OneHotEncoder (drop='first', handle_unknown='ignore')",
        "Save preprocessor + train/test arrays for model training",
        "Save feature names for importance analysis and Flask app",
    ]
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    print(f"\nNumeric features ({len(NUMERIC_FEATURES)}):")
    for col in NUMERIC_FEATURES:
        print(f"    - {col}")

    print(f"\nCategorical features ({len(CATEGORICAL_FEATURES)}):")
    for col in CATEGORICAL_FEATURES:
        print(f"    - {col}")

    # ------------------------------------------------------------------
    # 5.3 Run preprocessing pipeline
    # ------------------------------------------------------------------
    result = run_preprocessing(df)

    print_section("5.3 Train-Test Split Summary")
    print(f"Raw feature count        : {result['raw_features']}")
    print(f"Processed feature count  : {result['processed_features']} (after One-Hot Encoding)")
    print(f"Training rows            : {result['train_rows']:,}")
    print(f"Test rows                : {result['test_rows']:,}")
    print(f"Train churn rate         : {result['train_churn_rate_pct']:.2f}%")
    print(f"Test churn rate          : {result['test_churn_rate_pct']:.2f}%")
    print("Note: Stratified split keeps churn rate consistent across splits.")

    # ------------------------------------------------------------------
    # 5.4 Sample processed feature names
    # ------------------------------------------------------------------
    print_section("5.4 Sample Processed Feature Names (first 15)")
    for name in result["feature_names"][:15]:
        print(f"  - {name}")
    print(f"  ... ({result['processed_features']} total)")

    # ------------------------------------------------------------------
    # 5.5 Saved artifacts
    # ------------------------------------------------------------------
    print_section("5.5 Saved Artifacts")
    for key, path in result["paths"].items():
        print(f"  {key}: {path}")

    print_section("5.6 Why This Preprocessing Design?")
    rationale = [
        "StandardScaler helps Logistic Regression converge and compare coefficient magnitudes.",
        "OneHotEncoder converts categories to numeric form required by all three models.",
        "drop='first' avoids dummy variable trap for linear models.",
        "handle_unknown='ignore' allows Flask app to handle unseen categories safely.",
        "Fit-on-train-only prevents data leakage from test set statistics.",
        "Stratified split preserves ~26.5% churn rate in both train and test.",
    ]
    for i, point in enumerate(rationale, 1):
        print(f"  {i}. {point}")

    print_section("STEP 5 COMPLETE — Ready for Model Training (Step 6)")
    print("Review the artifacts above, then confirm to proceed to Step 6.")


if __name__ == "__main__":
    main()
