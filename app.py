import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------------
# Load Trained Pipeline
# -------------------------------
model = joblib.load("models/final_pipeline.pkl")

st.set_page_config(page_title="Credit Scoring System", layout="centered")

st.title("💳 Credit Scoring & Risk Assessment System")

st.markdown("Fill applicant details to evaluate creditworthiness.")

# -------------------------------
# User Inputs
# -------------------------------
age = st.slider("Age", 18, 75, 30)

loan_duration = st.slider("Loan Duration (Months)", 4, 60, 12)

credit_amount = st.number_input("Credit Amount", min_value=0, value=2000)

employment = st.selectbox(
    "Employment Duration",
    ["Unemployed", "<1 Year", "1-4 Years", "4+ Years"]
)

savings = st.selectbox(
    "Savings Account",
    ["Low", "Medium", "High"]
)

housing = st.selectbox(
    "Housing Type",
    ["Rent", "Own", "Free"]
)

# Business threshold control
threshold = st.slider("Approval Threshold", 0.1, 0.9, 0.5, 0.01)

# -------------------------------
# Convert to Model Format
# -------------------------------
employment_map = {
    "Unemployed": 0,
    "<1 Year": 1,
    "1-4 Years": 2,
    "4+ Years": 3
}

savings_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

housing_map = {
    "Rent": 1,
    "Own": 2,
    "Free": 3
}

input_data = {
    "laufkont": 1,
    "laufzeit": loan_duration,
    "moral": 2,
    "verw": 1,
    "hoehe": credit_amount,
    "sparkont": savings_map[savings],
    "beszeit": employment_map[employment],
    "rate": 2,
    "famges": 2,
    "buerge": 1,
    "wohnzeit": 2,
    "verm": 1,
    "alter": age,
    "weitkred": 1,
    "wohn": housing_map[housing],
    "bishkred": 1,
    "beruf": 2,
    "pers": 1,
    "telef": 1,
    "gastarb": 1,
    "credit_per_month": credit_amount / loan_duration
}

input_df = pd.DataFrame([input_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Credit Risk"):

    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Predicted Probability of Good Credit: {probability:.2f}")

    if probability >= threshold:
        st.success(f"✅ Approved (Good Credit Applicant)")
    else:
        st.error(f"⚠️ Rejected (High Risk Applicant)")