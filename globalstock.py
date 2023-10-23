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

def download_stock_data(ticker="AAPL", period="1d", interval="1m"):
    """
    Downloads stock data based on the provided ticker, period, and interval.

    :param ticker: Stock ticker. Default is "AAPL" for Apple.
    :param period: Data period. Default is "1d".
    :param interval: Data interval. Default is "1m".
    :return: Stock data in a DataFrame.
    """
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

if __name__ == "__main__":
    ticker = input("Enter the stock ticker (default: AAPL): ") or "AAPL"
    period = input("Enter the data period (default: 1d): ") or "1d"
    interval = input("Enter the data interval (default: 1m): ") or "1m"

    raw_data = download_stock_data(ticker, period, interval)
    
    # Display or use the final_data as needed
    print(final_data)
