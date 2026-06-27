"""Project paths and constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATA_FILE = DATA_RAW / "Telco-Customer-Churn.csv"
CLEANED_DATA_FILE = DATA_PROCESSED / "cleaned_telco_churn.csv"
ENGINEERED_DATA_FILE = DATA_PROCESSED / "engineered_telco_churn.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
TRAIN_TEST_FILE = DATA_PROCESSED / "train_test_data.joblib"
FEATURE_NAMES_FILE = DATA_PROCESSED / "feature_names.json"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# Saved model files
LIGHTGBM_MODEL_FILE = MODELS_DIR / "lightgbm_model.joblib"
LOGISTIC_REGRESSION_MODEL_FILE = MODELS_DIR / "logistic_regression_model.joblib"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest_model.joblib"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"

PRIMARY_MODEL_NAME = "lightgbm"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Modeling configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns excluded from the feature matrix
EXCLUDE_COLUMNS = [ID_COLUMN, TARGET_COLUMN, "ChurnNumeric"]

# Feature groups for ColumnTransformer
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend",
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

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]

# Service add-on columns (used for ServiceCount feature)
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Valid categorical levels (used during cleaning validation)
CONTRACT_TYPES = ("Month-to-month", "One year", "Two year")
INTERNET_SERVICE_TYPES = ("DSL", "Fiber optic", "No")
