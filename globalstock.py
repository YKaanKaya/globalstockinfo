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

def save_to_csv(dataframe, filename):
    """
    Saves the provided DataFrame to a CSV file.

    :param dataframe: Data to save.
    :param filename: Name of the CSV file.
    """
    dataframe.to_csv(filename)
    print(f"Data saved to {filename}.")

if __name__ == "__main__":
    # 1. User Input
    ticker = input("Enter the stock ticker (default: AAPL): ") or "AAPL"
    period = input("Enter the data period (default: 1d): ") or "1d"
    interval = input("Enter the data interval (default: 1m): ") or "1m"

    # 2. Download Data
    data = download_stock_data(ticker, period, interval)

    # 3. Display Data
    print(data)

    # 4. Option to Save
    save_option = input("Do you want to save the data to a CSV? (yes/no): ").lower()
    if save_option == "yes":
        filename = f"{ticker}_data.csv"
        save_to_csv(data, filename)
