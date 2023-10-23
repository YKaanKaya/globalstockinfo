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
import ticker_fetcher

def main():
    st.title("Stock Data Downloader")

    # User input for stock ticker and period
    ticker = st.text_input("Enter Stock Ticker:", "").upper()
    available_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.selectbox("Select Period:", available_periods, index=2)  # default to "1mo"

    if st.button("Download"):
        if not ticker:
            st.warning("Please enter a ticker.")
        else:
            try:
                data = yf.download(ticker, period=period)
                st.write(data)
                
                # Allow user to download the data
                csv = data.to_csv(index=True)
                b64 = b64encode(csv.encode()).decode()  # B64 encoding for downloading
                href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_{period}.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    from base64 import b64encode
    main()
