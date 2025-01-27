import streamlit as st
import h2o
from h2o.estimators import H2OXGBoostEstimator
import pandas as pd

# Initialize H2O
h2o.init()

# Load the saved model (path where you uploaded the model)
model_path = 'model/XGBoost_2_AutoML_1_20250127_170730'
model = h2o.load_model(model_path)

# Function for making predictions
def make_prediction(ticker, current_ratio, gross_margin, net_profit_margin, debt_to_equity, eps, free_cash_flow):
    # Create a dataframe for the new data point
    new_data = pd.DataFrame({
        'ticker': [ticker],
        'Current Ratio': [current_ratio],
        'Gross Margin': [gross_margin],
        'Net Profit Margin': [net_profit_margin],
        'Debt-to-Equity Ratio': [debt_to_equity],
        'Earnings Per Share (EPS)': [eps],
        'Free Cash Flow': [free_cash_flow]
    })
    
    # Convert to H2O frame
    new_data_h2o = h2o.H2OFrame(new_data)

    # Predict using the model
    prediction = model.predict(new_data_h2o)
    return prediction

# Streamlit user interface
st.title('Stock Evaluation Dashboard')

ticker = st.text_input("Enter Stock Ticker:")
current_ratio = st.number_input("Enter Current Ratio:")
gross_margin = st.number_input("Enter Gross Margin:")
net_profit_margin = st.number_input("Enter Net Profit Margin:")
debt_to_equity = st.number_input("Enter Debt-to-Equity Ratio:")
eps = st.number_input("Enter Earnings Per Share (EPS):")
free_cash_flow = st.number_input("Enter Free Cash Flow:")

if st.button("Get Recommendation"):
    # Get prediction from the model
    prediction = make_prediction(ticker, current_ratio, gross_margin, net_profit_margin, debt_to_equity, eps, free_cash_flow)

    # Display recommendation
    st.write(f"Recommendation: {prediction['predict'][0]}")
