import streamlit as st
import pandas as pd
from fmp_python import fmp

# Set your FMP API key here
API_KEY = 'j6kCIBjZa1pHewFjf7XaRDlslDxEFuof'  # Replace with your actual API key
fmp.set_api_key(API_KEY)

# Streamlit UI setup
st.title('Stock Fundamental Data Dashboard')

# User input for the stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT):')

# Fetch data from FMP API and display it
if ticker:
    try:
        # Fetch stock fundamental data
        stock_data = fmp.get_fundamentals(ticker)

        # Check if data exists for the given ticker
        if stock_data.empty:
            st.error(f"No data found for ticker: {ticker}")
        else:
            # Display stock fundamental data
            st.write(f"**Fundamental Data for {ticker}:**")
            st.dataframe(stock_data)

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

