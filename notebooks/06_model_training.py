"""
Step 6: Model Training
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/06_model_training.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    BEST_MODEL_FILE,
    LIGHTGBM_MODEL_FILE,
    LOGISTIC_REGRESSION_MODEL_FILE,
    PRIMARY_MODEL_NAME,
    RANDOM_FOREST_MODEL_FILE,
)
from train_models import build_models, run_training  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEP 6: MODEL TRAINING — IBM Telco Customer Churn")

    # ------------------------------------------------------------------
    # 6.1 Model selection rationale
    # ------------------------------------------------------------------
    print_section("6.1 Models Selected")
    models_info = {
        "LightGBM (Primary)": [
            "Gradient boosting — state-of-the-art for tabular data",
            "Handles non-linear relationships and feature interactions",
            "Fast training, strong performance on mixed feature types",
            "class_weight='balanced' addresses churn class imbalance",
        ],
        "Logistic Regression (Baseline)": [
            "Simple, interpretable linear baseline",
            "Industry standard for binary classification benchmarks",
            "Coefficients explain direction of feature influence",
        ],
        "Random Forest (Comparison)": [
            "Ensemble of decision trees — robust to outliers",
            "Captures non-linear patterns without boosting",
            "Good comparison point for tree-based vs boosting approaches",
        ],
    }
    for model_name, points in models_info.items():
        print(f"\n  {model_name}")
        for point in points:
            print(f"    • {point}")

    # ------------------------------------------------------------------
    # 6.2 Hyperparameters
    # ------------------------------------------------------------------
    print_section("6.2 Hyperparameters")
    for name, model in build_models().items():
        print(f"\n  {name}:")
        key_params = {
            k: v
            for k, v in model.get_params().items()
            if k
            in {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "num_leaves",
                "max_iter",
                "solver",
                "min_samples_split",
                "class_weight",
                "random_state",
            }
        }
        for k, v in key_params.items():
            print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # 6.3 Train all models
    # ------------------------------------------------------------------
    print_section("6.3 Training Models on Processed Training Set")
    result = run_training()

    print(f"Training samples   : {result['train_rows']:,}")
    print(f"Feature dimensions : {result['feature_count']}")
    print(f"Models trained     : {', '.join(result['models_trained'])}")

    print("\nTraining-set sanity check (not final evaluation — see Step 7):")
    for name, info in result["registry"]["models"].items():
        print(
            f"  {name:22s} | train accuracy: {info['train_accuracy']:.4f} | "
            f"predicted churn rate: {info['train_churn_rate_predicted']:.2%}"
        )

    # ------------------------------------------------------------------
    # 6.4 Saved model files
    # ------------------------------------------------------------------
    print_section("6.4 Saved Model Files")
    paths = {
        "LightGBM": LIGHTGBM_MODEL_FILE,
        "Logistic Regression": LOGISTIC_REGRESSION_MODEL_FILE,
        "Random Forest": RANDOM_FOREST_MODEL_FILE,
        f"Primary ({PRIMARY_MODEL_NAME})": BEST_MODEL_FILE,
        "Model Registry": result["registry_path"],
    }
    for label, path in paths.items():
        print(f"  {label:22s} → {path}")

    print_section("STEP 6 COMPLETE — Ready for Model Evaluation (Step 7)")
    print("Next: confusion matrix, accuracy, precision, recall, F1 on TEST set.")
    print("Confirm to proceed to Step 7.")


if __name__ == "__main__":
    main()
