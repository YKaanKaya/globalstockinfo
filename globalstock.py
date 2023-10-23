import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import base64

# Function to download stock data using yfinance
def download_stock_data(tickers, period, interval):
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by='ticker')
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Function to process the downloaded data and compute cumulative return and moving average
def process_data(data, period):
    try:
        # Rearranging the DataFrame
        portfolio = data.stack(level=0).reset_index().rename(columns={"level_1": "Symbol", "Date": "Datetime"})

        # Calculating cumulative returns
        portfolio['Cumulative Return'] = (portfolio['Close'] - portfolio.groupby('Symbol')['Close'].transform('first')) / portfolio.groupby('Symbol')['Close'].transform('first')

        # Calculating moving average based on the chosen period
        portfolio[f"MA-{period}"] = portfolio.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=int(period[:-1]), min_periods=1).mean())

        # Reordering the DataFrame columns
        columns_order = ["Symbol", "Datetime", "Open", "Close", "Cumulative Return", f"MA-{period}"]
        return portfolio[columns_order]

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# Title for the Streamlit app
st.title("Stock Data Downloader")

# Sidebar controls for user input
st.sidebar.header("Select Options")

# Accept multiple tickers from the user
tickers = st.sidebar.multiselect("Choose Tickers", ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT'], default=['AAPL', 'TSLA'])

# Period selection
period = st.sidebar.selectbox("Select Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=2)

# Interval selection
interval = st.sidebar.selectbox("Select Interval", ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index=8)

# Downloading and processing the data based on user selection
data = download_stock_data(tickers, period, interval)
if data is not None:
    processed_data = process_data(data, period)
    if processed_data is not None:
        # Display the processed data on the app
        st.write(processed_data)

# User-friendly instructions for downloading CSV
st.markdown("To download the displayed data as CSV:")
st.markdown("1. Click on the menu (three horizontal dots) on the top right of the data table.")
st.markdown("2. Click on 'Download CSV'.")

# A bit more about the app
st.markdown("""
**About the App:**
This app lets you select specific stock tickers, a desired period, and interval to fetch historical stock data using the yfinance library. It then computes the cumulative return and a moving average for the stocks and displays the results in a table. You can also download the table's contents as a CSV file.
""")
