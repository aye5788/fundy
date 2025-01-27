import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import h2o
from h2o.estimators import H2OXGBoostEstimator
from fmp import FinancialModelingPrepAPI

# Load trained model
model_path = 'model/XGBoost_2_AutoML_1_20250127_170730'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Setup FMP API
API_KEY = "j6kCIBjZa1pHewFjf7XaRDlslDxEFuof"
fmp_api = FinancialModelingPrepAPI(API_KEY)

# Streamlit interface for user input
st.title("Stock Prediction Based on Fundamental Analysis")
ticker = st.text_input("Enter stock ticker:", value="AAPL")

# Fetch fundamental data from FMP API
def fetch_fundamental_data(ticker):
    try:
        # Fetch Income Statement, Balance Sheet, and Cash Flow
        income_statement = fmp_api.get_income_statement(ticker)
        balance_sheet = fmp_api.get_balance_sheet(ticker)
        cash_flow = fmp_api.get_cash_flow(ticker)
        return income_statement, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None

# Preprocess data (Ensure the correct features used during training are extracted here)
def preprocess_data(income_statement, balance_sheet, cash_flow):
    # Example: Extract and preprocess key data points for prediction
    data = {
        "revenue": income_statement["revenue"],
        "gross_profit": income_statement["grossProfit"],
        "operating_income": income_statement["operatingIncome"],
        "net_income": income_statement["netIncome"],
        "total_assets": balance_sheet["totalAssets"],
        "total_liabilities": balance_sheet["totalLiabilities"],
        "shareholder_equity": balance_sheet["totalShareholderEquity"],
        "cash_flow_from_operations": cash_flow["operatingCashFlow"],
        # Add more features as per model training
    }
    return pd.DataFrame([data])

# Predict stock movement based on fundamental data
def predict_stock_movement(fundamental_data):
    prediction = model.predict(fundamental_data)
    return prediction

# Button to trigger prediction
if ticker:
    st.write(f"Fetching fundamental data for {ticker}...")
    income_statement, balance_sheet, cash_flow = fetch_fundamental_data(ticker)
    
    if income_statement and balance_sheet and cash_flow:
        st.write(f"Displaying fundamental data for {ticker}:")
        st.write("Income Statement:", income_statement)
        st.write("Balance Sheet:", balance_sheet)
        st.write("Cash Flow:", cash_flow)

        # Preprocess data
        processed_data = preprocess_data(income_statement, balance_sheet, cash_flow)

        # Make prediction
        prediction = predict_stock_movement(processed_data)
        st.write(f"Prediction: {prediction}")
    else:
        st.error("Unable to retrieve fundamental data.")
