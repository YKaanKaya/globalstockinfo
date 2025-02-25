import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpha Vantage API key from Streamlit secrets
api_key = st.secrets["A_KEY"]

# Cache TTL
CACHE_TTL = 3600  # 1 hour

def format_large_number(value):
    """Format large numbers with appropriate suffixes (M, B, T)."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

@st.cache_data(ttl=CACHE_TTL)
def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data with retry logic for resilience.
    """
    try:
        # Implement retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching historical data for {ticker} (Attempt {attempt+1})")
                # Fetch historical data from yfinance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if df.empty:
                    logger.warning(f"No data available for {ticker} in the selected date range.")
                    return None
                logger.info(f"Successfully fetched data for {ticker}")
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for {ticker}: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 2
                else:
                    raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_company_info(ticker):
    """
    Fetch company information with retry logic.
    """
    try:
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching company info for {ticker} (Attempt {attempt+1})")
                # Fetch company info from yfinance
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract relevant information
                result = {
                    'name': info.get('shortName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'fullTimeEmployees': info.get('fullTimeEmployees', 'N/A'),
                    'country': info.get('country', 'N/A'),
                    'marketCap': info.get('marketCap', 'N/A'),
                    'forwardPE': info.get('forwardPE', 'N/A'),
                    'trailingPE': info.get('trailingPE', 'N/A'),
                    'priceToBook': info.get('priceToBook', 'N/A'),
                    'dividendYield': info.get('dividendYield', 'N/A'),
                    'beta': info.get('beta', 'N/A'),
                    'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
                    'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
                    'longBusinessSummary': info.get('longBusinessSummary', 'N/A'),
                    'website': info.get('website', 'N/A'),
                    'logo_url': info.get('logo_url', 'N/A'),
                    'address1': info.get('address1', ''),
                    'city': info.get('city', ''),
                    'state': info.get('state', ''),
                    'zip': info.get('zip', ''),
                    'phone': info.get('phone', 'N/A'),
                }
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for company info {ticker}: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_esg_data(ticker):
    """
    Fetch ESG data for a company.
    """
    try:
        logger.info(f"Fetching ESG data for {ticker}")
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        if esg_data is None or esg_data.empty:
            logger.warning(f"No ESG data found for {ticker}")
            return None
        logger.info(f"Successfully fetched ESG data for {ticker}")
        return esg_data
    except Exception as e:
        logger.error(f"Error fetching ESG data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_competitors(ticker, max_competitors=5):
    """
    Get competitors based on industry classification.
    """
    try:
        logger.info(f"Finding competitors for {ticker}")
        # Get the industry of the selected ticker
        company_info = get_company_info(ticker)
        if not company_info or company_info['industry'] == 'N/A':
            logger.warning(f"Industry information not available for {ticker}")
            return []
        industry = company_info['industry']

        # Get the list of S&P 500 companies
        sp500_df = get_sp500_companies()
        if sp500_df is None or sp500_df.empty:
            logger.warning("Could not retrieve S&P 500 companies.")
            return []

        # Filter companies in the same industry
        competitors_df = sp500_df[sp500_df['Industry'] == industry]
        # Exclude the selected ticker
        competitors_df = competitors_df[competitors_df['Ticker'] != ticker]

        # Get the list of competitor tickers
        competitors = competitors_df['Ticker'].tolist()[:max_competitors]
        logger.info(f"Found {len(competitors)} competitors for {ticker}")
        return competitors
    except Exception as e:
        logger.error(f"Error fetching competitors for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def get_sp500_companies():
    """
    Fetch the list of S&P 500 companies from Wikipedia.
    """
    try:
        logger.info("Fetching S&P 500 companies list")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        df.columns = ['Ticker', 'Company', 'Sector', 'Industry']
        logger.info(f"Successfully fetched {len(df)} S&P 500 companies")
        return df
    except Exception as e:
        logger.error(f"Error fetching S&P 500 companies list: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def compare_performance(ticker, competitors, period_days=365):
    """
    Compare performance between ticker and competitors over a given period.
    """
    try:
        if not competitors:
            logger.warning("No competitors found for comparison.")
            return None
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        logger.info(f"Comparing performance of {ticker} with {len(competitors)} competitors")
        all_tickers = [ticker] + competitors
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=min(10, len(all_tickers))) as executor:
            futures = {}
            for t in all_tickers:
                futures[executor.submit(yf.download, t, start=start_date, end=end_date)] = t
            
            # Collect results
            data_dict = {}
            for future in futures:
                ticker_data = future.result()
                if not ticker_data.empty:
                    t = futures[future]
                    data_dict[t] = ticker_data['Adj Close']
        
        # Create DataFrame from collected data
        data = pd.DataFrame(data_dict)
        
        if data.empty:
            logger.warning("No competitor data available.")
            return None
            
        # Calculate returns
        returns = (data.pct_change() + 1).cumprod()
        logger.info("Successfully compared performance")
        return returns
    except Exception as e:
        logger.error(f"Error comparing performance: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_news(ticker, max_news=10):
    """
    Fetch latest news for a given ticker.
    """
    try:
        logger.info(f"Fetching news for {ticker}")
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            logger.warning(f"No news found for {ticker}")
            return []
        logger.info(f"Successfully fetched {len(news)} news items for {ticker}")
        return news[:max_news]
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def get_analyst_estimates(ticker):
    """
    Fetch analyst estimates from Alpha Vantage.
    """
    try:
        logger.info(f"Fetching analyst estimates for {ticker}")
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "ANALYST_ESTIMATES",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "analystEstimates" not in data:
            logger.warning(f"No analyst estimates data available for {ticker}")
            return None

        estimates = data["analystEstimates"]
        # Convert the estimates to a DataFrame
        estimates_df = pd.DataFrame(estimates)
        logger.info(f"Successfully fetched analyst estimates for {ticker}")
        return estimates_df
    except Exception as e:
        logger.error(f"Error fetching analyst estimates for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_income_statement(ticker):
    """
    Fetch income statement data from Alpha Vantage.
    """
    try:
        logger.info(f"Fetching income statement for {ticker}")
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            logger.warning(f"No income statement data available for {ticker}")
            return None

        income_statement = pd.DataFrame(data["annualReports"])
        income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
        logger.info(f"Successfully fetched income statement for {ticker}")
        return income_statement
    except Exception as e:
        logger.error(f"Error fetching income statement data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_balance_sheet(ticker):
    """
    Fetch balance sheet data from Alpha Vantage.
    """
    try:
        logger.info(f"Fetching balance sheet for {ticker}")
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            logger.warning(f"No balance sheet data available for {ticker}")
            return None

        balance_sheet = pd.DataFrame(data["annualReports"])
        balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
        logger.info(f"Successfully fetched balance sheet for {ticker}")
        return balance_sheet
    except Exception as e:
        logger.error(f"Error fetching balance sheet data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_cash_flow(ticker):
    """
    Fetch cash flow data from Alpha Vantage.
    """
    try:
        logger.info(f"Fetching cash flow for {ticker}")
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "CASH_FLOW",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            logger.warning(f"No cash flow data available for {ticker}")
            return None

        cash_flow = pd.DataFrame(data["annualReports"])
        cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
        logger.info(f"Successfully fetched cash flow for {ticker}")
        return cash_flow
    except Exception as e:
        logger.error(f"Error fetching cash flow data for {ticker}: {str(e)}")
        return None

def compute_analyst_consensus(estimates_df):
    """
    Compute analyst consensus from estimates DataFrame.
    """
    if estimates_df is None or estimates_df.empty:
        logger.warning("No data available for analyst consensus.")
        return None

    try:
        # Look for recommendation data
        if 'recommendationKey' in estimates_df.columns:
            recs = estimates_df['recommendationKey'].dropna()
            # Count the number of 'buy', 'hold', 'sell' recommendations
            buy_terms = ['buy', 'strong_buy']
            hold_terms = ['hold']
            sell_terms = ['sell', 'strong_sell']

            buy_count = recs[recs.isin(buy_terms)].count()
            hold_count = recs[recs.isin(hold_terms)].count()
            sell_count = recs[recs.isin(sell_terms)].count()

            total = buy_count + hold_count + sell_count
            if total == 0:
                return None

            consensus = {
                'Buy': int(buy_count),
                'Hold': int(hold_count),
                'Sell': int(sell_count)
            }
            return consensus
        else:
            logger.warning("No recommendation data found in analyst estimates.")
            return None
    except Exception as e:
        logger.error(f"Error computing analyst consensus: {str(e)}")
        return None

# Function to prefetch common stock data (can be called on startup)
def prefetch_common_stocks():
    """
    Prefetch data for commonly searched stocks to improve performance.
    """
    common_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Prefetching data for {len(common_tickers)} common stocks")
    
    with ThreadPoolExecutor(max_workers=len(common_tickers)) as executor:
        # Prefetch stock data
        for ticker in common_tickers:
            executor.submit(get_stock_data, ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            executor.submit(get_company_info, ticker)
    
    logger.info("Prefetching completed")
