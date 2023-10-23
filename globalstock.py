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

def download_stock_data(tickers=["AAPL"], period="1d", interval="1m"):
    all_data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, period=period, interval=interval)
        all_data[ticker] = stock_data
    return all_data

def main():
    st.title("Stock Data Downloader")

    # 1. User Input
    default_tickers = ["AAPL"]
    tickers = st.multiselect("Enter the stock tickers", default=default_tickers)

    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.selectbox("Select Data Period", valid_periods, index=0)

    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = st.selectbox("Select Data Interval", valid_intervals, index=0)

    # 2. Download and Display Data
    if st.button("Fetch Data"):
        with st.spinner("Fetching Data..."):
            data_dict = download_stock_data(tickers, period, interval)
            for ticker, data in data_dict.items():
                st.subheader(ticker)
                st.write(data)

        # 3. Option to Save
        if st.button("Download as CSV"):
            for ticker, data in data_dict.items():
                csv = data.to_csv(index=True)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_data.csv">Download {ticker} CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
