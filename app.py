import streamlit as st
import pandas as pd
import h2o
from h2o.estimators import H2OXGBoostEstimator
import requests
import json

# Initialize H2O cluster
h2o.init()

# Define the FMP API class to fetch financial data
class FinancialModelingPrepAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_income_statement(self, ticker):
        url = f"{self.base_url}/income-statement/{ticker}?apikey={self.api_key}"
        response = requests.get(url)
        return response.json()[0] if response.status_code == 200 else {}

    def get_balance_sheet(self, ticker):
        url = f"{self.base_url}/balance-sheet-statement/{ticker}?apikey={self.api_key}"
        response = requests.get(url)
        return response.json()[0] if response.status_code == 200 else {}

    def get_cash_flow(self, ticker):
        url = f"{self.base_url}/cash-flow-statement/{ticker}?apikey={self.api_key}"
        response = requests.get(url)
        return response.json()[0] if response.status_code == 200 else {}

# Function to make predictions using the trained model
def predict(model, input_data):
    h2o_input_data = h2o.H2OFrame(input_data)
    predictions = model.predict(h2o_input_data)
    return predictions

# Streamlit app UI
st.title("Stock Analysis and Prediction Dashboard")

# API Key for FMP (replace with your own key)
api_key = "j6kCIBjZa1pHewFjf7XaRDlslDxEFuof"
fmp_api = FinancialModelingPrepAPI(api_key)

# User input for ticker
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", "AAPL")

# Fetch fundamental data from FMP API
if ticker:
    income_statement = fmp_api.get_income_statement(ticker)
    balance_sheet = fmp_api.get_balance_sheet(ticker)
    cash_flow = fmp_api.get_cash_flow(ticker)

    if income_statement and balance_sheet and cash_flow:
        # Displaying the fundamental data for the user
        st.subheader("Income Statement")
        st.write(income_statement)

        st.subheader("Balance Sheet")
        st.write(balance_sheet)

        st.subheader("Cash Flow Statement")
        st.write(cash_flow)

        # Process data and make predictions based on trained model
        input_data = {
            'grossProfit': [income_statement.get('grossProfit')],
            'totalRevenue': [income_statement.get('totalRevenue')],
            'operatingIncome': [income_statement.get('operatingIncome')],
            'netIncome': [income_statement.get('netIncome')],
            'totalLiabilities': [balance_sheet.get('totalLiabilities')],
            'totalAssets': [balance_sheet.get('totalAssets')],
            'totalShareholderEquity': [balance_sheet.get('totalShareholderEquity')],
            'operatingCashflow': [cash_flow.get('operatingCashflow')],
            'capitalExpenditures': [cash_flow.get('capitalExpenditures')]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Load the trained model (replace path with correct one)
        model_path = "/path/to/your/model/folder/XGBoost_2_AutoML_1_20250127_170730"
        model = h2o.load_model(model_path)

        # Get prediction
        prediction = predict(model, input_df)

        # Display the recommendation based on the model prediction
        if prediction[0][0] > 0.5:
            st.subheader("Recommendation: **Buy**")
        else:
            st.subheader("Recommendation: **Avoid**")
    else:
        st.error(f"Unable to fetch data for ticker: {ticker}")

# Footer with instructions
st.sidebar.header("How it works")
st.sidebar.text("""
1. Enter a valid stock ticker (e.g., AAPL, MSFT).
2. View the fetched fundamental data for the stock.
3. Get a recommendation based on a trained machine learning model.
""")

