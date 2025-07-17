import streamlit as st
import pandas as pd
from joblib import load
from backend.gnn_predictor import predict_gnn

st.set_page_config(page_title="SmartMarketAI Dashboard", layout="wide")

st.title("🧠 SmartMarketAI Dashboard")
st.markdown("""This dashboard predicts conversions, suggests pricing, and detects fraud in ad campaigns using ML & GNN.""")

uploaded_file = st.file_uploader("📤 Upload your campaign data (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(df.head())

        with st.expander("🔍 Run Models", expanded=True):
            if st.button("🎯 Run Targeting Model"):
                model = load("../models/targeting_model.pkl")
                preds = model.predict(df.drop(columns=["converted"]))  # update if column is missing
                st.success("Predicted conversion likelihood for first 10 users:")
                st.write(preds[:10])

            if st.button("💸 Run Pricing Optimizer"):
                model = load("../models/pricing_model.pkl")
                preds = model.predict(df.drop(columns=["bid_price"]))  # update if column is missing
                st.success("Suggested bid prices for first 10 campaigns:")
                st.write(preds[:10])

            if st.button("🕵️‍♀️ Run Fraud Detection (Isolation Forest)"):
                model = load("../models/fraud_model.pkl")
                scores = model.decision_function(df)
                st.success("Anomaly Scores for Fraud Detection:")
                st.write(scores[:10])

            if st.button("🧠 Run GNN Fraud Detection"):
                preds = predict_gnn(df)
                st.success("GNN-Based Fraud Predictions (1 = Fraud):")
                st.write(preds[:10])

    except Exception as e:
        st.error(f"❌ Error loading or processing file: {e}")
else:
    st.info("Please upload a CSV file to get started.")