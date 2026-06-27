"""
Step 8: Feature Importance
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/08_feature_importance.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explainability import run_explainability  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEP 8: FEATURE IMPORTANCE — IBM Telco Customer Churn")

    # Run explainability pipeline
    result = run_explainability()
    importances_df = result["importances_df"]
    model_type = result["model_type"]

    # 8.1 Model details
    print_section("8.1 Explainability Context")
    print(f"Model Evaluated   : {model_type}")
    print(f"Total Features    : {len(importances_df)}")

    # 8.2 Display top 10 features
    print_section("8.2 Top 10 Most Influential Features")
    print(f"{'Rank':4s} | {'Feature':32s} | {'Raw Importance':14s} | {'Cumulative %'}")
    print("-" * 70)
    
    total_imp = importances_df["importance"].sum()
    cumulative_pct = 0.0
    
    for i, row in importances_df.head(10).iterrows():
        rank = i + 1
        pct = (row["importance"] / total_imp) * 100 if total_imp > 0 else 0.0
        cumulative_pct += pct
        print(
            f"{rank:<4d} | "
            f"{row['feature_clean']:32s} | "
            f"{row['importance']:14.4f} | "
            f"{cumulative_pct:.2f}%"
        )

    # 8.3 Business Interpretation
    print_section("8.3 Key Business Takeaways (Drivers of Churn)")
    
    # Extract top features to dynamically explain
    top_feats = importances_df["feature"].head(5).tolist()
    
    interpretations = {
        "num__IsMonthToMonth": "1. Month-to-month contracts are a massive risk factor. Customers with no long-term commitment churn at a dramatically higher rate.",
        "num__tenure": "2. Tenure is a strong stabilizer. Long-term loyalty reduces the probability of churn significantly.",
        "num__TotalCharges": "3. Total Charges represents customer lifetime value; higher values indicate long-term clients who are highly stable.",
        "num__AvgMonthlySpend": "4. Average Monthly Spend indicates billing intensity; customers who spend more on average are highly price-sensitive.",
        "num__MonthlyCharges": "5. Monthly Charges indicate billing intensity; high monthly charges drive customer dissatisfaction and exit.",
        "num__IsFiber": "6. Fiber Optic Internet is associated with high churn, possibly due to pricing pressure, technical issues, or competitors' promotions.",
        "num__IsElectronicCheck": "7. Electronic Check payment method has a strong positive correlation with churn, indicating transactional friction compared to auto-pay.",
        "num__IsAutoPay": "8. Automatic Payments (Credit Card/Bank Transfer) create payment stability and lower transactional churn.",
    }
    
    printed_count = 0
    for feat in top_feats:
        if feat in interpretations:
            print(f"  • {interpretations[feat]}")
            printed_count += 1
            
    if printed_count < 3:
        # Fallback general interpretation
        print("  • Contract types (Month-to-month) and tenure remain the strongest predictors of churn.")
        print("  • High monthly charges and specific service types (like Fiber Optic) represent key risk clusters.")
        print("  • Transactional friction (e.g. Electronic Checks vs Auto-pay) plays a noticeable role in customer exit.")

    # 8.4 Output files
    print_section("8.4 Saved Figures & Reports")
    print(f"  Feature Importance Chart : {result['feature_importance_plot']}")
    print(f"  Feature Importance CSV   : {result['feature_importance_csv']}")

    print_section("STEP 8 COMPLETE — Ready for Business Insights (Step 9)")
    print("Next: Calculate customer risk scores, financial impacts, and targeted retention rules.")
    print("Confirm to proceed to Step 9.")


if __name__ == "__main__":
    main()
