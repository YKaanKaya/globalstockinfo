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

def download_stock_data(tickers, period, interval):
    try:
        # Download data
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def process_data(Portfolio):
    try:
        portfolio = Portfolio.copy().reset_index().rename(index=str, columns={"index": "Datetime"})
        portfolio['Return'] = (portfolio['Close'] - portfolio['Open']) / portfolio['Open']

        # Move 'Symbol' column to the front
        cols = list(portfolio.columns)
        cols.insert(0, cols.pop(cols.index('Symbol')))
        portfolio = portfolio[cols]

        return portfolio
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

def main():
    st.title("Stock Data Viewer")

    # 1. User Inputs
    tickers = st.text_input("Enter Ticker(s) (comma-separated for multiple)", "AAPL, TSLA").split(',')
    period = st.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    interval = st.selectbox("Select Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])

    # 2. Download, Process and Display Data
    with st.spinner("Fetching Data..."):
        raw_data = download_stock_data([ticker.strip().upper() for ticker in tickers], period, interval)
        processed_data = process_data(raw_data)
        st.write(processed_data)

if __name__ == "__main__":
    main()
