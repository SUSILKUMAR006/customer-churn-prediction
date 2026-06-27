"""Streamlit dashboard application for Telecom Customer Churn Prediction and Targeted Retention.

Compatible with local execution and Streamlit Cloud deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Setup path to include project source
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    BEST_MODEL_FILE,
    PREPROCESSOR_FILE,
    REPORTS_DIR,
)
from feature_engineering import engineer_features
from preprocessing import prepare_features_target

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    /* Global style overrides */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
    }
    
    /* Custom Sidebar Card Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Metrics panel card custom backgrounds */
    .kpi-card-blue {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #c2410c, #f97316);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }
    .kpi-card-cyan {
        background: linear-gradient(135deg, #0e7490, #06b6d4);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }
    
    .kpi-title {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 !important;
    }
    .kpi-value {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin: 5px 0 !important;
    }
    .kpi-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.8rem !important;
        margin: 0 !important;
    }
    
    /* Playbook Card Styling */
    .playbook-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }
    .playbook-title {
        color: #3b82f6;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        margin-top: 0px;
    }
    .playbook-detail {
        font-size: 0.9rem;
        margin-bottom: 4px;
        color: #94a3b8;
    }
    
    /* Cohort Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: #fca5a5;
        color: #991b1b;
    }
    .badge-medium {
        background-color: #fde047;
        color: #854d0e;
    }
    .badge-low {
        background-color: #86efac;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


# --- RESOURCE CACHING ---
@st.cache_resource
def load_ml_resources():
    """Load machine learning models and pipelines."""
    preprocessor = None
    model = None
    if PREPROCESSOR_FILE.exists():
        preprocessor = joblib.load(PREPROCESSOR_FILE)
    if BEST_MODEL_FILE.exists():
        model = joblib.load(BEST_MODEL_FILE)
    return preprocessor, model


@st.cache_data
def load_data_resources():
    """Load scored customers list and business metrics summary."""
    scored_customers = None
    insights_summary = None
    
    # Check if files exist in reports figures directory
    scored_csv_path = REPORTS_DIR / "scored_customers.csv"
    if scored_csv_path.exists():
        scored_customers = pd.read_csv(scored_csv_path)
        
    insights_json_path = REPORTS_DIR / "business_insights_summary.json"
    if insights_json_path.exists():
        with open(insights_json_path, "r", encoding="utf-8") as f:
            insights_summary = json.load(f)
            
    return scored_customers, insights_summary


def load_all_resources():
    """Verify and load all resources, running the business insights pipeline if needed."""
    preprocessor, model = load_ml_resources()
    scored_customers, insights_summary = load_data_resources()
    
    # If analytics data is missing, attempt to run the business insights generation script
    if preprocessor is None or model is None or scored_customers is None or insights_summary is None:
        try:
            from business_insights import run_business_insights
            run_business_insights()
            
            # Reset caches to fetch the newly generated files
            st.cache_resource.clear()
            st.cache_data.clear()
            
            preprocessor, model = load_ml_resources()
            scored_customers, insights_summary = load_data_resources()
        except Exception as e:
            st.sidebar.error(f"Failed to generate resources automatically: {e}")
            
    return preprocessor, model, scored_customers, insights_summary


# Initialize and load resources
preprocessor, model, scored_customers, insights_summary = load_all_resources()


# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
        <div style='width: 32px; height: 32px; background: linear-gradient(135deg, #3b82f6, #06b6d4); border-radius: 8px;'></div>
        <h2 style='margin: 0; color: #ffffff;'>RetinaAI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    tab_selection = st.radio(
        label="Go to",
        options=["📊 Analytics Dashboard", "🔮 Single Churn Predictor", "📞 High-Risk Outreach"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model info card
    st.markdown("""
    <div style='background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155;'>
        <p style='margin: 0; font-size: 0.8rem; color: #94a3b8;'>ACTIVE CLASSIFIER</p>
        <p style='margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 700; color: #f8fafc;'>Random Forest Classifier</p>
        <p style='margin: 2px 0 0 0; font-size: 0.85rem; color: #22c55e;'>● Test F1-Score: 0.6291</p>
    </div>
    """, unsafe_allow_html=True)


# --- TAB 1: ANALYTICS DASHBOARD ---
if tab_selection == "📊 Analytics Dashboard":
    st.title("Telecom Customer Churn Dashboard")
    st.caption("Real-time churn risk scoring and targeted customer retention playbooks")
    
    if insights_summary is None:
        st.error("Analytics metrics are not available. Please run the notebook pipeline to generate data.")
    else:
        # KPIs Row
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            total_base = insights_summary.get("total_customers", 7043)
            st.markdown(f"""
            <div class="kpi-card-blue">
                <p class="kpi-title">Total Active Base</p>
                <p class="kpi-value">{total_base:,}</p>
                <p class="kpi-subtitle">Active Accounts Scored</p>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            rev_at_risk = insights_summary.get("monthly_revenue_at_risk", 188871)
            pct_at_risk = insights_summary.get("monthly_revenue_at_risk_pct", 41.41)
            st.markdown(f"""
            <div class="kpi-card-orange">
                <p class="kpi-title">Monthly Revenue at Risk</p>
                <p class="kpi-value">${rev_at_risk:,.2f}</p>
                <p class="kpi-subtitle">{pct_at_risk:.2f}% of total billing</p>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi3:
            potential_savings = insights_summary.get("potential_monthly_savings", 37774)
            st.markdown(f"""
            <div class="kpi-card-cyan">
                <p class="kpi-title">Potential Campaign Savings</p>
                <p class="kpi-value">${potential_savings:,.2f}</p>
                <p class="kpi-subtitle">Assuming 20% retention rate</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Analysis Grid
        col_cohorts, col_drivers = st.columns(2)
        
        with col_cohorts:
            st.subheader("Churn Risk Cohorts Breakdown")
            st.caption("Customers grouped by machine learning prediction probabilities")
            
            cohorts = insights_summary.get("cohorts", {})
            high_cohort = cohorts.get("High Risk (>=70%)", {"count": 1582, "pct": 22.46, "revenue": 122500})
            med_cohort = cohorts.get("Medium Risk (30-70%)", {"count": 1862, "pct": 26.44, "revenue": 132860})
            low_cohort = cohorts.get("Low Risk (<30%)", {"count": 3599, "pct": 51.10, "revenue": 200756})
            
            # High Risk Progress Bar
            st.markdown(f"**High Risk (>=70%)** - **{high_cohort['count']:,}** customers ({high_cohort['pct']:.2f}%)")
            st.progress(high_cohort['pct'] / 100.0)
            st.markdown(f"<p style='font-size: 0.85rem; color: #94a3b8; margin-top: -10px; margin-bottom: 15px;'>Monthly revenue: ${high_cohort['revenue']:,.2f}</p>", unsafe_allow_html=True)
            
            # Medium Risk Progress Bar
            st.markdown(f"**Medium Risk (30-70%)** - **{med_cohort['count']:,}** customers ({med_cohort['pct']:.2f}%)")
            st.progress(med_cohort['pct'] / 100.0)
            st.markdown(f"<p style='font-size: 0.85rem; color: #94a3b8; margin-top: -10px; margin-bottom: 15px;'>Monthly revenue: ${med_cohort['revenue']:,.2f}</p>", unsafe_allow_html=True)
            
            # Low Risk Progress Bar
            st.markdown(f"**Low Risk (<30%)** - **{low_cohort['count']:,}** customers ({low_cohort['pct']:.2f}%)")
            st.progress(low_cohort['pct'] / 100.0)
            st.markdown(f"<p style='font-size: 0.85rem; color: #94a3b8; margin-top: -10px; margin-bottom: 15px;'>Monthly revenue: ${low_cohort['revenue']:,.2f}</p>", unsafe_allow_html=True)
            
        with col_drivers:
            st.subheader("Top Churn Drivers & Importance")
            st.caption("Most influential factors determining customer churn risk")
            
            # Display Feature Importance Image if exists
            feat_imp_img_path = REPORTS_DIR / "feature_importance.png"
            if feat_imp_img_path.exists():
                st.image(str(feat_imp_img_path), use_container_width=True)
            else:
                st.markdown("""
                1. **Short tenure with the company**: Customers in their first 6 months have an overall churn rate of 52.94%.
                2. **Month-to-month contracts**: Lack of long-term contract lock-in correlates with a 42.71% churn rate.
                3. **Fiber optic internet service**: Fiber users churn at 41.89%, suggesting high price-sensitivity on premium tiers.
                4. **Electronic check payment friction**: Paying manually via e-check shows a 45.29% churn rate compared to auto-pay (~15%).
                """)


# --- TAB 2: SINGLE CHURN PREDICTOR ---
elif tab_selection == "🔮 Single Churn Predictor":
    st.title("Predict Customer Churn Risk")
    st.caption("Input customer features to calculate risk and retrieve targeted retention playbooks")
    
    if preprocessor is None or model is None:
        st.error("Model assets not loaded. Run pipeline scripts to generate `preprocessor.joblib` and `best_model.joblib` first.")
    else:
        # Prediction Form split in 2 columns
        with st.form("churn_prediction_form"):
            col_dem, col_srv = st.columns(2)
            
            with col_dem:
                st.subheader("Demographics & Contract Details")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                
            with col_srv:
                st.subheader("Service Details & Billing")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                
                # Dynamic MultipleLines option based on phone service selection
                ml_options = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
                multiple_lines = st.selectbox("Multiple Lines", ml_options)
                
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                
                # Dynamic internet add-on options
                if internet_service == "No":
                    online_security = "No internet service"
                    online_backup = "No internet service"
                    device_protection = "No internet service"
                    tech_support = "No internet service"
                    streaming_tv = "No internet service"
                    streaming_movies = "No internet service"
                    
                    st.text_input("Online Security", online_security, disabled=True)
                    st.text_input("Online Backup", online_backup, disabled=True)
                    st.text_input("Device Protection", device_protection, disabled=True)
                    st.text_input("Tech Support", tech_support, disabled=True)
                    st.text_input("Streaming TV", streaming_tv, disabled=True)
                    st.text_input("Streaming Movies", streaming_movies, disabled=True)
                else:
                    online_security = st.selectbox("Online Security", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
                    
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
                total_charges_input = st.text_input("Total Charges ($) [Leave empty for auto-calculate]", value="")
                
            submit_btn = st.form_submit_button("Calculate Churn Risk", use_container_width=True)
            
        if submit_btn:
            # Parse numeric senior citizen representation
            sc_num = 1 if senior_citizen == "Yes" else 0
            
            # Parse TotalCharges
            if total_charges_input.strip() == "":
                total_charges = monthly_charges * tenure if tenure > 0 else 0.0
            else:
                try:
                    total_charges = float(total_charges_input)
                except ValueError:
                    st.error("Invalid entry for Total Charges. Defaulting to Monthly Charges * tenure.")
                    total_charges = monthly_charges * tenure if tenure > 0 else 0.0

            # Construct input raw dataframe
            customer_dict = {
                "customerID": ["PRED-TEMP"],
                "gender": [gender],
                "SeniorCitizen": [sc_num],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
                "Churn": ["No"] # Placeholder
            }
            
            df_raw = pd.DataFrame(customer_dict)
            
            # Execute feature engineering
            df_engineered, _ = engineer_features(df_raw)
            
            # Separate features X
            X, _ = prepare_features_target(df_engineered)
            
            # Transform and predict
            try:
                X_processed = preprocessor.transform(X)
                prob = float(model.predict_proba(X_processed)[0, 1])
                pred = int(model.predict(X_processed)[0])
                
                # Set risk cohorts
                if prob >= 0.70:
                    cohort = "High Risk (>=70%)"
                    badge_class = "badge-high"
                    color_hex = "#ef4444"
                elif prob >= 0.30:
                    cohort = "Medium Risk (30-70%)"
                    badge_class = "badge-medium"
                    color_hex = "#f59e0b"
                else:
                    cohort = "Low Risk (<30%)"
                    badge_class = "badge-low"
                    color_hex = "#10b981"
                    
                # Layout results
                st.markdown("---")
                st.subheader("Analysis & Score Card")
                
                col_score, col_metrics = st.columns(2)
                
                with col_score:
                    st.markdown(f"""
                    <div style='background-color: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid #334155; text-align: center;'>
                        <p style='margin: 0; font-size: 1.1rem; color: #94a3b8;'>CHURN PROBABILITY</p>
                        <p style='margin: 10px 0; font-size: 3.5rem; font-weight: 800; color: {color_hex};'>{prob * 100:.1f}%</p>
                        <span class="badge {badge_class}">{cohort}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_metrics:
                    st.markdown("##### Engineered Feature Insights")
                    
                    avg_monthly = df_engineered["AvgMonthlySpend"].iloc[0]
                    srv_count = int(df_engineered["ServiceCount"].iloc[0])
                    ratio = df_engineered["ChargeIncreaseRatio"].iloc[0]
                    tenure_grp = df_engineered["TenureGroup"].iloc[0]
                    
                    st.markdown(f"""
                    * **Avg Monthly Spend**: `${avg_monthly:.2f}` (TotalCharges / tenure)
                    * **Active Services Count**: `{srv_count} / 6` (Selected add-ons)
                    * **Billing Charge Increase Ratio**: `{ratio:.2f}` (MonthlyCharges / AvgMonthlySpend)
                    * **Tenure Group**: `{tenure_grp}`
                    """)
                    
                # Playbook Generation
                st.markdown("### Targeted Retention Playbooks")
                playbooks_triggered = []
                
                if df_engineered["IsMonthToMonth"].iloc[0] == 1:
                    playbooks_triggered.append({
                        "title": "Playbook A: Proactive Contract Conversion",
                        "trigger": "Customer is on a high-risk Month-to-month contract.",
                        "action": "Offer a 15% discount on Monthly Charges in exchange for a 1-year contract, or a 25% discount for a 2-year contract conversion.",
                        "benefit": "Locks in the customer and immediately reduces structural churn probability."
                    })
                    
                if df_engineered["IsFiber"].iloc[0] == 1:
                    playbooks_triggered.append({
                        "title": "Playbook B: Fiber Optic Tech Support Bundle",
                        "trigger": "Customer is on a premium Fiber Optic line (statistically higher churn).",
                        "action": "Proactively attach a free 3-month Tech Support and Online Security bundle to improve quality-of-service perception.",
                        "benefit": "Addresses quality/technical churn drivers on premium lines."
                    })
                    
                if df_engineered["IsElectronicCheck"].iloc[0] == 1:
                    playbooks_triggered.append({
                        "title": "Playbook C: Auto-Pay Enrollment Drive",
                        "trigger": "Customer pays via Electronic Check (high transactional friction).",
                        "action": "Offer a one-time $10 bill credit to enroll in Automatic Direct Debit or Credit Card billing.",
                        "benefit": "Removes monthly manual payment friction, eliminating behavioral churn triggers."
                    })
                    
                if df_engineered["IsNewCustomer"].iloc[0] == 1:
                    playbooks_triggered.append({
                        "title": "Playbook D: Customer Success Onboarding Call",
                        "trigger": "Customer is in the critical first 6 months of onboarding (tenure <= 6).",
                        "action": "Trigger a customer success outreach call to ensure successful setup, address early friction, and walk through features.",
                        "benefit": "Provides high-touch engagement during the highest risk segment window."
                    })
                    
                if not playbooks_triggered:
                    playbooks_triggered.append({
                        "title": "General Loyalty Campaign",
                        "trigger": "No high-risk structural triggers detected.",
                        "action": "Send a standard quarterly thank-you email or educational newsletter showcasing new features.",
                        "benefit": "Maintains passive brand awareness and customer relationship health."
                    })
                    
                for pb in playbooks_triggered:
                    st.markdown(f"""
                    <div class="playbook-card" style="border-left: 4px solid {color_hex};">
                        <p class="playbook-title">{pb['title']}</p>
                        <p class="playbook-detail"><strong>Trigger:</strong> {pb['trigger']}</p>
                        <p class="playbook-detail"><strong>Action:</strong> {pb['action']}</p>
                        <p class="playbook-detail"><strong>Benefit:</strong> {pb['benefit']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as ex:
                st.error(f"Error executing model: {ex}")


# --- TAB 3: HIGH-RISK OUTREACH ---
elif tab_selection == "📞 High-Risk Outreach":
    st.title("Customer Outreach Queue")
    st.caption("Outreach playbooks and contact queue for the highest-risk churn cohorts")
    
    if scored_customers is None:
        st.error("No scored customer list available. Run pipeline to score customer database.")
    else:
        # Search & Filter
        col_search, col_cohort = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("Search Customer ID", "").strip().upper()
        with col_cohort:
            cohort_filter = st.selectbox("Cohort Filter", ["All", "High Risk (>=70%)", "Medium Risk (30-70%)", "Low Risk (<30%)"])
            
        # Apply filters
        filtered_df = scored_customers
        if search_query:
            filtered_df = filtered_df[filtered_df["customerID"].str.upper().str.contains(search_query)]
        if cohort_filter != "All":
            filtered_df = filtered_df[filtered_df["ChurnRiskCohort"] == cohort_filter]
            
        # Select customer for detailed view
        if filtered_df.empty:
            st.warning("No customer records match your filters.")
        else:
            st.markdown(f"**Found {len(filtered_df):,} matching customers**")
            
            # Select customer to see details
            selected_id = st.selectbox("Select Customer to view detailed Playbook", filtered_df["customerID"].values)
            
            if selected_id:
                row = filtered_df[filtered_df["customerID"] == selected_id].iloc[0]
                
                # Prob and badge color
                p = row["ChurnProbability"]
                if p >= 0.70:
                    badge_style = "badge-high"
                    c_hex = "#ef4444"
                elif p >= 0.30:
                    badge_style = "badge-medium"
                    c_hex = "#f59e0b"
                else:
                    badge_style = "badge-low"
                    c_hex = "#10b981"
                    
                st.markdown("---")
                
                # Split details and playbooks
                col_det, col_play = st.columns([1, 2])
                
                with col_det:
                    st.markdown(f"#### Customer: `{row['customerID']}`")
                    st.markdown(f"""
                    <div style='background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #94a3b8;'>Risk Score</p>
                        <p style='margin: 5px 0; font-size: 1.8rem; font-weight: 800; color: {c_hex};'>{p * 100:.1f}%</p>
                        <span class="badge {badge_style}">{row['ChurnRiskCohort']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    **Contract & Demographics:**
                    * **Tenure**: `{row['tenure']} months`
                    * **Contract**: `{row['Contract']}`
                    * **Payment**: `{row['PaymentMethod']}`
                    * **Monthly Charges**: `${row['MonthlyCharges']:.2f}`
                    * **Total Charges**: `${row['TotalCharges']:.2f}`
                    """)
                    
                with col_play:
                    st.markdown("#### Dynamic Retention Playbooks")
                    
                    # Resolve playbook triggers
                    playbooks = []
                    
                    if row.get("IsMonthToMonth", 0) == 1:
                        playbooks.append({
                            "title": "Playbook A: Proactive Contract Conversion",
                            "trigger": "Customer is on a high-risk Month-to-month contract.",
                            "action": "Offer a 15% discount on Monthly Charges in exchange for a 1-year contract, or a 25% discount for a 2-year contract conversion.",
                            "benefit": "Locks in the customer and immediately reduces structural churn probability."
                        })
                        
                    if row.get("IsFiber", 0) == 1:
                        playbooks.append({
                            "title": "Playbook B: Fiber Optic Tech Support Bundle",
                            "trigger": "Customer is on a premium Fiber Optic line (statistically higher churn).",
                            "action": "Proactively attach a free 3-month Tech Support and Online Security bundle to improve quality-of-service perception.",
                            "benefit": "Addresses quality/technical churn drivers on premium lines."
                        })
                        
                    if row.get("IsElectronicCheck", 0) == 1:
                        playbooks.append({
                            "title": "Playbook C: Auto-Pay Enrollment Drive",
                            "trigger": "Customer pays via Electronic Check (high transactional friction).",
                            "action": "Offer a one-time $10 bill credit to enroll in Automatic Direct Debit or Credit Card billing.",
                            "benefit": "Removes monthly manual payment friction, eliminating behavioral churn triggers."
                        })
                        
                    if row.get("IsNewCustomer", 0) == 1:
                        playbooks.append({
                            "title": "Playbook D: Customer Success Onboarding Call",
                            "trigger": "Customer is in the critical first 6 months of onboarding (tenure <= 6).",
                            "action": "Trigger a customer success outreach call to ensure successful setup, address early friction, and walk through features.",
                            "benefit": "Provides high-touch engagement during the highest risk segment window."
                        })
                        
                    if not playbooks:
                        playbooks.append({
                            "title": "General Loyalty Campaign",
                            "trigger": "No high-risk structural triggers detected.",
                            "action": "Send a standard quarterly thank-you email or educational newsletter showcasing new features.",
                            "benefit": "Maintains passive brand awareness and customer relationship health."
                        })
                        
                    for pb in playbooks:
                        st.markdown(f"""
                        <div class="playbook-card" style="border-left: 4px solid {c_hex};">
                            <p class="playbook-title">{pb['title']}</p>
                            <p class="playbook-detail"><strong>Trigger:</strong> {pb['trigger']}</p>
                            <p class="playbook-detail"><strong>Action:</strong> {pb['action']}</p>
                            <p class="playbook-detail"><strong>Benefit:</strong> {pb['benefit']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Full Scored Database Queue")
            
            # Format dataframe display columns
            display_df = filtered_df[[
                "customerID", "tenure", "Contract", "InternetService", "MonthlyCharges", "ChurnProbability", "ChurnRiskCohort"
            ]].copy()
            
            display_df["ChurnProbability"] = display_df["ChurnProbability"].apply(lambda val: f"{val * 100:.1f}%")
            display_df["MonthlyCharges"] = display_df["MonthlyCharges"].apply(lambda val: f"${val:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
