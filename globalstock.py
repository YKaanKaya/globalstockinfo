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
        if "Symbol" not in Portfolio.columns:
            Portfolio = Portfolio.stack(level=0).reset_index().rename(columns={"level_1": "Symbol"}).set_index("Datetime")
        
        # Calculate daily return
        Portfolio['Daily Return'] = (Portfolio['Close'] - Portfolio['Open']) / Portfolio['Open']
        
        # Calculate cumulative return
        Portfolio['Cumulative Return'] = (1 + Portfolio['Daily Return']).cumprod() - 1
        
        # Move 'Symbol' column to the front
        cols = list(Portfolio.columns)
        cols.insert(0, cols.pop(cols.index('Symbol')))
        Portfolio = Portfolio[cols]

        return Portfolio
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
        
        # Calculate Moving Average
        period_mapping = {"1d": 1, "5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}
        processed_data[f"Moving Avg ({period})"] = processed_data.groupby('Symbol')['Close'].transform(lambda x: x.rolling(period_mapping.get(period, 1)).mean())
        
        st.write(processed_data)

if __name__ == "__main__":
    main()
