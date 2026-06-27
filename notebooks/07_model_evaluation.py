"""
Step 7: Model Evaluation
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/07_model_evaluation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import run_evaluation  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEP 7: MODEL EVALUATION — IBM Telco Customer Churn")

    # Run evaluation
    result = run_evaluation()
    test_metrics = result["test_metrics"]
    best_model_name = result["best_model_name"]
    best_model_metrics = result["best_model_metrics"]

    # 7.1 Display comparison table
    print_section("7.1 Model Comparison on Test Set")
    
    header = f"{'Model':22s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1-Score':9s} | {'ROC-AUC':9s}"
    print(header)
    print("-" * len(header))
    
    for name, metrics in test_metrics.items():
        display_name = name.replace("_", " ").title()
        row = (
            f"{display_name:22s} | "
            f"{metrics['accuracy']:.4f}    | "
            f"{metrics['precision']:.4f}    | "
            f"{metrics['recall']:.4f}    | "
            f"{metrics['f1']:.4f}    | "
            f"{metrics['roc_auc']:.4f}"
        )
        print(row)

    # 7.2 Explain best model selection
    print_section("7.2 Best Model Selection")
    display_best_name = best_model_name.replace("_", " ").title()
    print(f"Selected Model : {display_best_name}")
    print(f"Selection Rule : Highest Test F1-Score")
    print(f"Test F1-Score  : {best_model_metrics['f1']:.4f}")
    print(f"Test Recall    : {best_model_metrics['recall']:.4f}")
    print(f"Test Precision : {best_model_metrics['precision']:.4f}")
    print(f"Test ROC-AUC   : {best_model_metrics['roc_auc']:.4f}")
    print(f"\nNote: The best model has been copied to 'models/best_model.joblib'.")

    # 7.3 Exported plots
    print_section("7.3 Exported Plots & Figures")
    print(f"  Confusion Matrices : {result['confusion_matrix_plot']}")
    print(f"  Evaluation Curves  : {result['evaluation_curves_plot']}")
    print(f"  Model Registry     : {result['registry_path']}")

    print_section("STEP 7 COMPLETE — Ready for Feature Importance (Step 8)")
    print("Next: Analyze top feature importances of the selected model.")
    print("Confirm to proceed to Step 8.")


if __name__ == "__main__":
    main()
