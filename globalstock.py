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
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

def save_to_csv(dataframe, filename):
    dataframe.to_csv(filename)
    print(f"Data saved to {filename}.")

def get_valid_input(prompt, accepted_values):
    user_input = input(prompt).lower()
    while user_input not in accepted_values:
        print(f"Invalid input. Accepted values: {', '.join(accepted_values)}")
        user_input = input(prompt).lower()
    return user_input

if __name__ == "__main__":
    # 1. User Input
    ticker = input("Enter the stock ticker (default: AAPL): ") or "AAPL"

    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = get_valid_input("Enter the data period (default: 1d): ", valid_periods) or "1d"

    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = get_valid_input("Enter the data interval (default: 1m): ", valid_intervals) or "1m"

    # 2. Download Data
    data = download_stock_data(ticker, period, interval)

    # 3. Display Data
    print(data)

    # 4. Option to Save
    save_option = input("Do you want to save the data to a CSV? (yes/no): ").lower()
    if save_option == "yes":
        filename = f"{ticker}_data.csv"
        save_to_csv(data, filename)
