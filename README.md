# Customer Churn Prediction

End-to-end machine learning project for predicting telecom customer churn using the **IBM Telco Customer Churn** dataset.

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original CSV (Telco-Customer-Churn.csv)
│   └── processed/              # Cleaned & engineered datasets
├── notebooks/
│   ├── 01_data_understanding.ipynb / .py
│   ├── 02_data_cleaning.ipynb / .py
│   ├── 03_eda.ipynb / .py
│   ├── 04_feature_engineering.ipynb / .py
│   ├── 05_preprocessing.ipynb / .py
│   └── 06_model_training.ipynb / .py
├── src/
│   ├── config.py               # Paths and constants
│   ├── data_loader.py          # Dataset loading
│   ├── data_cleaning.py        # Cleaning pipeline
│   ├── eda.py                  # EDA plots & summaries
│   ├── feature_engineering.py  # Feature creation pipeline
│   ├── preprocessing.py        # Encoding, scaling, train-test split
│   └── train_models.py         # Model training pipeline
├── models/                     # Saved preprocessor & ML models
├── app/                        # Flask web application (later stages)
├── reports/figures/            # EDA & evaluation plots
├── docs/                       # PPT outline, architecture diagrams
└── requirements.txt
```

## Setup

```bash
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Dataset

Place `Telco-Customer-Churn.csv` in `data/raw/` or download from:
- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM GitHub mirror](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Pipeline Progress

| Step | Stage | Status |
|------|-------|--------|
| 1 | Data Understanding | ✅ Complete |
| 2 | Data Cleaning | ✅ Complete |
| 3 | EDA | ✅ Complete |
| 4 | Feature Engineering | ✅ Complete |
| 5 | Preprocessing | ✅ Complete |
| 6 | Model Training | ✅ Complete |
| 7 | Model Evaluation | ⏳ Pending |
| 8 | Feature Importance | ⏳ Pending |
| 9 | Business Insights | ⏳ Pending |
| 10 | Retention Recommendations | ⏳ Pending |
| 11 | Flask Web App | ⏳ Pending |
| 12 | Documentation & PPT | ⏳ Pending |

## Step 1 — Run Data Understanding

```bash
python notebooks/01_data_understanding.py
# or open notebooks/01_data_understanding.ipynb
```

## Step 2 — Run Data Cleaning

```bash
python notebooks/02_data_cleaning.py
# or open notebooks/02_data_cleaning.ipynb
```

Output: `data/processed/cleaned_telco_churn.csv`

## Step 3 — Run EDA

```bash
python notebooks/03_eda.py
# or open notebooks/03_eda.ipynb
```

Output: plots in `reports/figures/` and `reports/figures/eda_churn_summary.csv`

## Step 4 — Run Feature Engineering

```bash
python notebooks/04_feature_engineering.py
# or open notebooks/04_feature_engineering.ipynb
```

Output: `data/processed/engineered_telco_churn.csv`

## Step 5 — Run Preprocessing

```bash
python notebooks/05_preprocessing.py
# or open notebooks/05_preprocessing.ipynb
```

Outputs:
- `models/preprocessor.joblib`
- `data/processed/train_test_data.joblib`
- `data/processed/feature_names.json`

## Step 6 — Run Model Training

```bash
python notebooks/06_model_training.py
# or open notebooks/06_model_training.ipynb
```

Outputs:
- `models/lightgbm_model.joblib`
- `models/logistic_regression_model.joblib`
- `models/random_forest_model.joblib`
- `models/best_model.joblib` (LightGBM — primary)
- `models/model_registry.json`
