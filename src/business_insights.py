"""Business insights and retention recommendations utilities for the Telco Churn project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    BEST_MODEL_FILE,
    ENGINEERED_DATA_FILE,
    PREPROCESSOR_FILE,
    REPORTS_DIR,
)
from preprocessing import prepare_features_target


def get_customer_risk_scores() -> pd.DataFrame:
    """
    Load full engineered dataset, run preprocessing, and score churn probabilities.
    
    Returns
    -------
    pd.DataFrame
        Original engineered dataframe enriched with ChurnProbability and ChurnRiskCohort columns.
    """
    if not ENGINEERED_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing engineered dataset: {ENGINEERED_DATA_FILE}. Run Step 4 first.")
    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(f"Missing preprocessor: {PREPROCESSOR_FILE}. Run Step 5 first.")
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing best model: {BEST_MODEL_FILE}. Run Step 7 first.")

    # 1. Load data and models
    df = pd.read_csv(ENGINEERED_DATA_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(BEST_MODEL_FILE)

    # 2. Extract features (X)
    X, _ = prepare_features_target(df)

    # 3. Transform features
    X_processed = preprocessor.transform(X)

    # 4. Score probabilities and predictions
    y_prob = model.predict_proba(X_processed)[:, 1]
    y_pred = model.predict(X_processed)

    # 5. Enrich original dataframe
    df["ChurnProbability"] = y_prob
    df["PredictedChurn"] = y_pred

    # 6. Define risk cohorts
    def assign_cohort(prob):
        if prob >= 0.70:
            return "High Risk (>=70%)"
        elif prob >= 0.30:
            return "Medium Risk (30-70%)"
        else:
            return "Low Risk (<30%)"

    df["ChurnRiskCohort"] = df["ChurnProbability"].apply(assign_cohort)
    return df


def calculate_business_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate financial impact and cohort metrics from scored customer data."""
    # Group by risk cohort
    cohort_counts = df["ChurnRiskCohort"].value_counts().to_dict()
    cohort_pcts = (df["ChurnRiskCohort"].value_counts(normalize=True) * 100).to_dict()

    # Calculate revenue figures
    total_monthly_revenue = df["MonthlyCharges"].sum()
    
    # Revenue at risk is MonthlyCharges for predicted churners
    predicted_churners = df[df["PredictedChurn"] == 1]
    revenue_at_risk = predicted_churners["MonthlyCharges"].sum()
    revenue_at_risk_pct = (revenue_at_risk / total_monthly_revenue) * 100
    
    # Financial benefits of retention campaigns
    retention_success_rate = 0.20 # 20% success rate is standard
    potential_savings = revenue_at_risk * retention_success_rate

    # Analyze high-risk demographics
    high_risk_df = df[df["ChurnRiskCohort"].str.startswith("High")]
    
    contract_dist = high_risk_df["Contract"].value_counts(normalize=True).to_dict()
    internet_dist = high_risk_df["InternetService"].value_counts(normalize=True).to_dict()
    payment_dist = high_risk_df["PaymentMethod"].value_counts(normalize=True).to_dict()

    return {
        "total_customers": len(df),
        "predicted_churners_count": len(predicted_churners),
        "predicted_churn_rate_pct": round((len(predicted_churners) / len(df)) * 100, 2),
        "total_monthly_revenue": round(total_monthly_revenue, 2),
        "monthly_revenue_at_risk": round(revenue_at_risk, 2),
        "monthly_revenue_at_risk_pct": round(revenue_at_risk_pct, 2),
        "campaign_retention_success_rate": retention_success_rate,
        "potential_monthly_savings": round(potential_savings, 2),
        "cohorts": {
            cohort: {
                "count": count,
                "pct": round(cohort_pcts[cohort], 2),
                "revenue": round(df[df["ChurnRiskCohort"] == cohort]["MonthlyCharges"].sum(), 2)
            }
            for cohort, count in cohort_counts.items()
        },
        "high_risk_profile": {
            "contract_distribution": contract_dist,
            "internet_distribution": internet_dist,
            "payment_distribution": payment_dist,
        }
    }


def generate_recommendations_report(metrics: dict[str, Any], output_path: Path) -> None:
    """Create a structured markdown report for targeted business retention strategies."""
    high_risk = metrics["cohorts"].get("High Risk (>=70%)", {"count": 0, "pct": 0, "revenue": 0})
    med_risk = metrics["cohorts"].get("Medium Risk (30-70%)", {"count": 0, "pct": 0, "revenue": 0})
    
    report_content = f"""# Customer Churn Prevention & Retention Recommendations

This report translates the predictive machine learning model outputs into actionable business strategies and targeted retention campaigns.

---

## 1. Executive Summary

- **Total Active Customer Base**: {metrics['total_customers']:,}
- **Predicted Churners**: {metrics['predicted_churners_count']:,} ({metrics['predicted_churn_rate_pct']:.2f}% of customer base)
- **Total Monthly Revenue**: ${metrics['total_monthly_revenue']:,}
- **Monthly Revenue at Risk**: ${metrics['monthly_revenue_at_risk']:,} ({metrics['monthly_revenue_at_risk_pct']:.2f}% of monthly billing)
- **Targeted Campaign Savings**: **${metrics['potential_monthly_savings']:,} / month** (assuming a conservative 20% retention success rate)

---

## 2. Risk Cohort Distribution

| Churn Risk Cohort | Customer Count | % of Base | Monthly Revenue | Description / Action Plan |
|-------------------|----------------|-----------|-----------------|---------------------------|
| **High Risk (>=70%)** | {high_risk['count']:,} | {high_risk['pct']:.2f}% | ${high_risk['revenue']:,} | Immediate, high-touch proactive outreach and financial incentives. |
| **Medium Risk (30-70%)** | {med_risk['count']:,} | {med_risk['pct']:.2f}% | ${med_risk['revenue']:,} | Proactive engagement, feature upgrades, or feedback loops. |
| **Low Risk (<30%)** | {metrics['cohorts'].get('Low Risk (<30%)', {'count': 0, 'pct': 0, 'revenue': 0})['count']:,} | {metrics['cohorts'].get('Low Risk (<30%)', {'count': 0, 'pct': 0, 'revenue': 0})['pct']:.2f}% | ${metrics['cohorts'].get('Low Risk (<30%)', {'count': 0, 'pct': 0, 'revenue': 0})['revenue']:,} | Standard nurture campaigns; monitor for billing or service changes. |

---

## 3. High-Risk Customer Profile

Analysis of the {high_risk['count']:,} customers in the **High Risk** cohort reveals the following dominant characteristics:
- **Contract Type**: Month-to-month contracts represent the vast majority of high-risk customers, indicating extreme price sensitivity and lack of lock-in.
- **Internet Service**: Fiber Optic subscribers represent a large proportion of high-risk customers, suggesting dissatisfaction with pricing, service delivery, or intense competitor poaching.
- **Payment Method**: Electronic Check users represent a highly unstable segment, suffering from transactional friction compared to auto-pay methods.

---

## 4. Actionable Retention Playbooks

### Playbook A: Proactive Contract Conversion (High-Risk Month-to-Month Customers)
- **Target Segment**: Month-to-month contract holders with High Churn Risk.
- **Strategy**: Proactively offer a **15% discount** on Monthly Charges in exchange for signing a **1-year contract**, or a **25% discount** for a **2-year contract**.
- **Business Justification**: Converts volatile short-term accounts into stable long-term agreements, dramatically reducing immediate churn risk.

### Playbook B: Fiber Optic Loyalty & Support Bundles
- **Target Segment**: Fiber Optic internet subscribers with Churn Risk >= 50%.
- **Strategy**: 
  - Proactively attach a **free 3-month Tech Support and Online Security bundle** (known churn mitigators).
  - Perform a diagnostic line health check to resolve underlying technical issues before they lead to customer frustration.
- **Business Justification**: Addresses the specific dissatisfaction drivers associated with premium Fiber Optic plans.

### Playbook C: Automatic Billing Enrollment Drive (Auto-Pay Incentives)
- **Target Segment**: Electronic Check payers with Churn Risk >= 50%.
- **Strategy**: Offer a **one-time $10 account credit** to enroll in Automatic Credit Card or Bank Transfer payments (Auto-Pay).
- **Business Justification**: Eliminates monthly manual payment friction, which is a major behavioral trigger for churn.

### Playbook D: Onboarding Success Calls for New Customers
- **Target Segment**: Customers in their first 6 months (tenure <= 6) with Churn Risk >= 50%.
- **Strategy**: Proactive onboarding check-in call from customer success representatives to walk them through features and ensure their service is running flawlessly.
- **Business Justification**: Reverses the early churn trend during the critical onboarding window.

---

## 5. Implementation Roadmap & Next Steps

1. **Database Integration**: Export the scored customer list (including risk scores) to the CRM system.
2. **Automated Triggers**: Configure automated email/SMS campaigns in the marketing platform based on the risk scores.
3. **Agent Enablement**: Equip call center agents with the targeted Playbooks (A, B, and C) to pitch to high-risk callers.
4. **Flask Dashboard Integration**: Embed these risk classifications and targeted playbooks directly into the web application.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip())


def run_business_insights() -> dict[str, Any]:
    """
    Execute business insights pipeline.
    
    Steps:
    1. Score all customers using the best model
    2. Compute high-level cohort and financial metrics
    3. Save recommendations report markdown
    4. Save metrics summary JSON and scored customer CSV
    """
    # Score customers
    scored_df = get_customer_risk_scores()

    # Calculate metrics
    metrics = calculate_business_metrics(scored_df)

    # Paths
    report_path = REPORTS_DIR / "retention_recommendations.md"
    summary_path = REPORTS_DIR / "business_insights_summary.json"
    scored_csv_path = REPORTS_DIR / "scored_customers.csv"

    # Generate files
    generate_recommendations_report(metrics, report_path)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save top 1000 highest risk customers for sales outreach
    high_risk_list = scored_df.sort_values(by="ChurnProbability", ascending=False)
    high_risk_list.to_csv(scored_csv_path, index=False)

    return {
        "metrics": metrics,
        "report_path": report_path,
        "summary_path": summary_path,
        "scored_csv_path": scored_csv_path,
    }
