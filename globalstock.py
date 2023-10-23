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

def download_stock_data(tickers=["AAPL", "TSLA"], period="1d", interval="1m"):
    all_data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, period=period, interval=interval)
        stock_data['Symbol'] = ticker  # Add a 'Symbol' column to differentiate data
        all_data[ticker] = stock_data[['Open', 'Close', 'Symbol']]

    combined_data = pd.concat(all_data.values(), axis=0)  # Combine all ticker data
    return combined_data

def process_data(Portfolio):
    try:
        portfolio = Portfolio.copy().reset_index().rename(index=str, columns={"index": "Datetime"})
        portfolio['Return'] = (portfolio['Close'] - portfolio['Open']) / portfolio['Open']
        return portfolio
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

def main():
    st.title("Stock Data Downloader")

    # Default values
    default_tickers = ["AAPL", "TSLA"]
    default_period = "1d"
    default_interval = "1m"

    # 1. User Input
    tickers = st.text_input("Enter the stock tickers (comma-separated)", value=",".join(default_tickers)).split(',')

    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.selectbox("Select Data Period", valid_periods, index=valid_periods.index(default_period))

    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = st.selectbox("Select Data Interval", valid_intervals, index=valid_intervals.index(default_interval))

    # 2. Download, Process and Display Data
    with st.spinner("Fetching Data..."):
        raw_data = download_stock_data([ticker.strip().upper() for ticker in tickers], period, interval)
        processed_data = process_data(raw_data)
        st.write(processed_data)

    # 3. Option to Save
    if st.button("Download as CSV"):
        csv = processed_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
