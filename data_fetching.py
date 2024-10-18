# data_fetching.py

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from utils import api_key

@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data using yfinance."""
    try:
        # Ensure tickers is a list if multiple tickers are provided
        if isinstance(tickers, str):
            tickers_list = [tickers]
        else:
            tickers_list = tickers

        # Fetch historical data from yfinance
        df = yf.download(tickers_list, start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data available for {tickers} in the selected date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {tickers}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    """Fetch company information using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extract relevant information
        company_info = {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'fullTimeEmployees': info.get('fullTimeEmployees', 'N/A'),
            'country': info.get('country', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'forwardPE': info.get('forwardPE', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A'),
            'longBusinessSummary': info.get('longBusinessSummary', 'N/A'),
            'website': info.get('website', 'N/A'),
            'address1': info.get('address1', ''),
            'city': info.get('city', ''),
            'state': info.get('state', ''),
            'zip': info.get('zip', ''),
            'phone': info.get('phone', 'N/A'),
        }
        return company_info
    except Exception as e:
        st.error(f"Error fetching company info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_esg_data(ticker):
    """Fetch ESG data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        if esg_data is None or esg_data.empty:
            st.warning(f"No ESG data found for {ticker}")
            return None
        return esg_data
    except Exception as e:
        st.error(f"Error fetching ESG data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_sp500_companies():
    """Fetch the list of S&P 500 companies from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        df.columns = ['Ticker', 'Company', 'Sector', 'Industry']
        return df
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies list: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_competitors(ticker):
    """Identify competitors within the same industry from S&P 500 list."""
    try:
        # Get the industry of the selected ticker
        company_info = get_company_info(ticker)
        if not company_info or company_info['industry'] == 'N/A':
            st.warning(f"Industry information not available for {ticker}")
            return []
        industry = company_info['industry']

        # Get the list of S&P 500 companies
        sp500_df = get_sp500_companies()
        if sp500_df is None or sp500_df.empty:
            st.warning("Could not retrieve S&P 500 companies.")
            return []

        # Filter companies in the same industry
        competitors_df = sp500_df[sp500_df['Industry'] == industry]
        # Exclude the selected ticker
        competitors_df = competitors_df[competitors_df['Ticker'] != ticker]

        # Get the list of competitor tickers
        competitors = competitors_df['Ticker'].tolist()[:5]  # Return top 5 competitors
        return competitors
    except Exception as e:
        st.error(f"Error fetching competitors for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_news(ticker):
    """Fetch latest news articles for the ticker."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_analyst_estimates(ticker):
    """Fetch analyst estimates data from Alpha Vantage or yfinance as fallback."""
    try:
        # Attempt to fetch from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "ANALYST_ESTIMATES",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "analystEstimates" in data:
            estimates = pd.DataFrame(data["analystEstimates"])
            if "recommendationKey" in estimates.columns:
                return estimates
            else:
                st.warning(f"'recommendationKey' not found in Alpha Vantage data for {ticker}.")
        else:
            st.warning(f"No analyst estimates data available for {ticker} from Alpha Vantage.")

        # If Alpha Vantage data is not available, fetch from yfinance
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            return recommendations
        else:
            st.warning(f"No analyst recommendations available for {ticker} from yfinance.")
            return None
    except Exception as e:
        st.error(f"Error fetching analyst estimates for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_income_statement(ticker):
    """Fetch income statement data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch income statement data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" in data and data["annualReports"]:
            income_statement = pd.DataFrame(data["annualReports"])
            income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
            return income_statement
        else:
            st.warning(f"No income statement data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            financials = stock.financials
            if financials is None or financials.empty:
                st.error(f"No income statement data available for {ticker} from yfinance.")
                return None
            income_statement = financials.transpose().reset_index()
            income_statement.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            return income_statement
    except Exception as e:
        st.error(f"Error fetching income statement data for {ticker} from Alpha Vantage: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            if financials is None or financials.empty:
                st.error(f"No income statement data available for {ticker} from yfinance.")
                return None
            income_statement = financials.transpose().reset_index()
            income_statement.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            return income_statement
        except Exception as y:
            st.error(f"Error fetching income statement data for {ticker} from yfinance: {str(y)}")
            return None

@st.cache_data(ttl=3600)
def get_balance_sheet(ticker):
    """Fetch balance sheet data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch balance sheet data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"No balance sheet data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                st.error(f"No balance sheet data available for {ticker} from yfinance.")
                return None
            balance_sheet = balance_sheet.transpose().reset_index()
            balance_sheet.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
        else:
            balance_sheet = pd.DataFrame(data["annualReports"])
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
    except Exception as e:
        st.error(f"Error fetching balance sheet data for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                st.error(f"No balance sheet data available for {ticker} from yfinance.")
                return None
            balance_sheet = balance_sheet.transpose().reset_index()
            balance_sheet.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
        except Exception as y:
            st.error(f"Error fetching balance sheet data for {ticker} from yfinance: {str(y)}")
            return None

@st.cache_data(ttl=3600)
def get_cash_flow(ticker):
    """Fetch cash flow data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch cash flow data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "CASH_FLOW",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"No cash flow data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow
            if cash_flow.empty:
                st.error(f"No cash flow data available for {ticker} from yfinance.")
                return None
            cash_flow = cash_flow.transpose().reset_index()
            cash_flow.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
        else:
            cash_flow = pd.DataFrame(data["annualReports"])
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
    except Exception as e:
        st.error(f"Error fetching cash flow data for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow
            if cash_flow.empty:
                st.error(f"No cash flow data available for {ticker} from yfinance.")
                return None
            cash_flow = cash_flow.transpose().reset_index()
            cash_flow.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
        except Exception as y:
            st.error(f"Error fetching cash flow data for {ticker} from yfinance: {str(y)}")
            return None
