"""
Step 9 & 10: Business Insights & Retention Recommendations
IBM Telco Customer Churn Dataset

Run from project root:
    python notebooks/09_business_insights.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from business_insights import run_business_insights  # noqa: E402


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("STEPS 9 & 10: BUSINESS INSIGHTS & RETENTION — IBM Telco Churn")

    # Run insights pipeline
    result = run_business_insights()
    metrics = result["metrics"]

    # 9.1 Executive Financial Summary
    print_section("9.1 Executive Financial Summary")
    print(f"Total Customer Base       : {metrics['total_customers']:,}")
    print(f"Predicted Churners        : {metrics['predicted_churners_count']:,} ({metrics['predicted_churn_rate_pct']:.2f}%)")
    print(f"Total Monthly Billing     : ${metrics['total_monthly_revenue']:,}")
    print(f"Monthly Revenue at Risk   : ${metrics['monthly_revenue_at_risk']:,} ({metrics['monthly_revenue_at_risk_pct']:.2f}%)")
    print(f"Targeted Campaign savings : ${metrics['potential_monthly_savings']:,} / month (assuming 20% success)")

    # 9.2 Churn Risk Cohorts
    print_section("9.2 Churn Risk Cohorts")
    print(f"{'Risk Cohort':22s} | {'Count':8s} | {'% Base':8s} | {'Monthly Revenue'}")
    print("-" * 65)
    
    for cohort, info in metrics["cohorts"].items():
        print(
            f"{cohort:22s} | "
            f"{info['count']:<8,d} | "
            f"{info['pct']:.2f}%   | "
            f"${info['revenue']:,}"
        )

    # 10.1 Key Retention Playbooks
    print_section("10.1 Key Retention Playbooks & Recommendations")
    playbooks = [
        "Playbook A: Proactive Contract Conversion (Month-to-Month High-Risk)",
        "  -> Offer 15% discount for 1-year contract or 25% for 2-year contract conversion.",
        "Playbook B: Fiber Optic Loyalty & Support Bundles",
        "  -> Attach 3-month free Technical Support & Security bundle to mitigate price/service churn.",
        "Playbook C: Auto-Pay Enrollment Drive",
        "  -> Offer $10 one-time credit to enroll in Automatic Credit/Bank transfer billing (resolves friction).",
        "Playbook D: Onboarding Success Calls for New Customers",
        "  -> Call customers in first 6 months (tenure <= 6) with Churn Risk >= 50% to ensure satisfaction.",
    ]
    for line in playbooks:
        print(line)

    # 10.2 Saved Reports
    print_section("10.2 Saved Reports & Output Lists")
    print(f"  Retention Playbook MD   : {result['report_path']}")
    print(f"  Insights Summary JSON   : {result['summary_path']}")
    print(f"  Scored Customer CSV     : {result['scored_csv_path']}")
    print(f"\nNote: Scored Customer CSV is sorted by churn probability descending (ready for outreach).")

    print_section("STEPS 9 & 10 COMPLETE — All Analytic Stages Finalized")
    print("Next: Proceed to Step 11 for the Flask Web Application dashboard.")


if __name__ == "__main__":
    main()
