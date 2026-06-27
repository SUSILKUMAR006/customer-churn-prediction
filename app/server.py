"""Flask backend server for the Customer Churn Prediction dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    BEST_MODEL_FILE,
    PREPROCESSOR_FILE,
    REPORTS_DIR,
)
from feature_engineering import engineer_features
from preprocessing import prepare_features_target

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load models and artifacts globally
preprocessor = None
model = None
scored_customers = None
insights_summary = None

def load_resources():
    global preprocessor, model, scored_customers, insights_summary
    
    # Load model and preprocessor
    if PREPROCESSOR_FILE.exists():
        preprocessor = joblib.load(PREPROCESSOR_FILE)
    if BEST_MODEL_FILE.exists():
        model = joblib.load(BEST_MODEL_FILE)
        
    # Load scored customers
    scored_csv_path = REPORTS_DIR / "scored_customers.csv"
    if scored_csv_path.exists():
        scored_customers = pd.read_csv(scored_csv_path)
        
    # Load insights summary
    insights_json_path = REPORTS_DIR / "business_insights_summary.json"
    if insights_json_path.exists():
        with open(insights_json_path, "r", encoding="utf-8") as f:
            insights_summary = json.load(f)

# Initial load
load_resources()


@app.route("/")
def index():
    """Serve the single-page dashboard UI."""
    return render_template("index.html")


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return high-level risk cohort and financial metrics."""
    global insights_summary
    if insights_summary is None:
        load_resources()
    
    if insights_summary is None:
        return jsonify({"error": "Analytics metrics are not generated yet. Run the pipeline first."}), 500
        
    return jsonify(insights_summary)


@app.route("/api/customers", methods=["GET"])
def get_customers():
    """
    Return a paginated, searchable list of scored customers.
    Query params:
      - search: string (matches customerID)
      - page: int (default 1)
      - limit: int (default 10)
    """
    global scored_customers
    if scored_customers is None:
        load_resources()
        
    if scored_customers is None:
        return jsonify({"error": "Scored customers list is not generated yet. Run the pipeline first."}), 500

    search = request.args.get("search", "").strip().upper()
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))

    df_filtered = scored_customers
    if search:
        df_filtered = scored_customers[scored_customers["customerID"].str.upper().str.contains(search)]

    total_records = len(df_filtered)
    total_pages = max(1, (total_records + limit - 1) // limit)
    page = min(max(1, page), total_pages)
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    records = df_filtered.iloc[start_idx:end_idx].to_dict(orient="records")
    
    # Return formatted response
    return jsonify({
        "records": records,
        "page": page,
        "limit": limit,
        "total_records": total_records,
        "total_pages": total_pages
    })


@app.route("/api/predict", methods=["POST"])
def predict_churn():
    """
    Receive single customer features, engineer them, and predict churn probability.
    """
    global preprocessor, model
    if preprocessor is None or model is None:
        load_resources()
        
    if preprocessor is None or model is None:
        return jsonify({"error": "Model or preprocessor is not loaded. Run the pipeline first."}), 500

    try:
        data = request.json
        if not data:
            return jsonify({"error": "Empty request payload."}), 400

        # Extract features and supply defaults if missing
        tenure = int(data.get("tenure", 0))
        monthly_charges = float(data.get("MonthlyCharges", 0.0))
        
        # Impute TotalCharges if missing or zero for existing customers
        total_charges = data.get("TotalCharges", None)
        if total_charges is None or str(total_charges).strip() == "":
            total_charges = monthly_charges * tenure if tenure > 0 else 0.0
        else:
            total_charges = float(total_charges)

        # Construct single-row DataFrame matching raw format
        customer_dict = {
            "customerID": ["PRED-TEMP"],
            "gender": [data.get("gender", "Female")],
            "SeniorCitizen": [int(data.get("SeniorCitizen", 0))],
            "Partner": [data.get("Partner", "No")],
            "Dependents": [data.get("Dependents", "No")],
            "tenure": [tenure],
            "PhoneService": [data.get("PhoneService", "Yes")],
            "MultipleLines": [data.get("MultipleLines", "No")],
            "InternetService": [data.get("InternetService", "No")],
            "OnlineSecurity": [data.get("OnlineSecurity", "No")],
            "OnlineBackup": [data.get("OnlineBackup", "No")],
            "DeviceProtection": [data.get("DeviceProtection", "No")],
            "TechSupport": [data.get("TechSupport", "No")],
            "StreamingTV": [data.get("StreamingTV", "No")],
            "StreamingMovies": [data.get("StreamingMovies", "No")],
            "Contract": [data.get("Contract", "Month-to-month")],
            "PaperlessBilling": [data.get("PaperlessBilling", "Yes")],
            "PaymentMethod": [data.get("PaymentMethod", "Mailed check")],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Churn": ["No"] # Placeholder
        }

        df_raw = pd.DataFrame(customer_dict)

        # Run feature engineering
        df_engineered, _ = engineer_features(df_raw)

        # Separate features X
        X, _ = prepare_features_target(df_engineered)

        # Run preprocessor transformation
        X_processed = preprocessor.transform(X)

        # Predict probability
        prob = float(model.predict_proba(X_processed)[0, 1])
        pred = int(model.predict(X_processed)[0])

        # Assign risk cohort
        if prob >= 0.70:
            cohort = "High Risk (>=70%)"
            color_class = "risk-high"
        elif prob >= 0.30:
            cohort = "Medium Risk (30-70%)"
            color_class = "risk-medium"
        else:
            cohort = "Low Risk (<30%)"
            color_class = "risk-low"

        # Generate targeted playbooks based on triggers
        playbooks = []
        
        # Playbook A: Contract Conversion
        if df_engineered["IsMonthToMonth"].iloc[0] == 1:
            playbooks.append({
                "title": "Playbook A: Proactive Contract Conversion",
                "trigger": "Customer is on a high-risk Month-to-month contract.",
                "action": "Offer a 15% discount on Monthly Charges in exchange for a 1-year contract, or a 25% discount for a 2-year contract conversion.",
                "benefit": "Locks in the customer and immediately reduces structural churn probability."
            })
            
        # Playbook B: Fiber Optic Loyalty
        if df_engineered["IsFiber"].iloc[0] == 1:
            playbooks.append({
                "title": "Playbook B: Fiber Optic Tech Support Bundle",
                "trigger": "Customer is on a premium Fiber Optic line (statistically higher churn).",
                "action": "Proactively attach a free 3-month Tech Support and Online Security bundle to improve quality-of-service perception.",
                "benefit": "Addresses quality/technical churn drivers on premium lines."
            })
            
        # Playbook C: Automatic Billing Enrollment
        if df_engineered["IsElectronicCheck"].iloc[0] == 1:
            playbooks.append({
                "title": "Playbook C: Auto-Pay Enrollment Drive",
                "trigger": "Customer pays via Electronic Check (high transactional friction).",
                "action": "Offer a one-time $10 bill credit to enroll in Automatic Direct Debit or Credit Card billing.",
                "benefit": "Removes monthly manual payment friction, eliminating behavioral churn triggers."
            })
            
        # Playbook D: Onboarding Check-in
        if df_engineered["IsNewCustomer"].iloc[0] == 1:
            playbooks.append({
                "title": "Playbook D: Customer Success Onboarding Call",
                "trigger": "Customer is in the critical first 6 months of onboarding (tenure <= 6).",
                "action": "Trigger a customer success outreach call to ensure successful setup, address early friction, and walk through features.",
                "benefit": "Provides high-touch engagement during the highest risk segment window."
            })

        # Fallback if no specific playbook triggered
        if not playbooks:
            playbooks.append({
                "title": "General Loyalty Campaign",
                "trigger": "No high-risk structural triggers detected.",
                "action": "Send a standard quarterly thank-you email or educational newsletter showcasing new features.",
                "benefit": "Maintains passive brand awareness and customer relationship health."
            })

        return jsonify({
            "probability": round(prob, 4),
            "prediction": pred,
            "cohort": cohort,
            "color_class": color_class,
            "engineered_features": {
                "AvgMonthlySpend": round(float(df_engineered["AvgMonthlySpend"].iloc[0]), 2),
                "ServiceCount": int(df_engineered["ServiceCount"].iloc[0]),
                "ChargeIncreaseRatio": round(float(df_engineered["ChargeIncreaseRatio"].iloc[0]), 3),
                "TenureGroup": str(df_engineered["TenureGroup"].iloc[0]),
            },
            "playbooks": playbooks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Ensure all directories exist
    app.run(host="127.0.0.1", port=5000, debug=True)
