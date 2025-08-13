import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache TTL
CACHE_TTL = 3600  # 1 hour

class EnhancedDataFetcher:
    """Enhanced data fetcher using only yfinance with comprehensive data extraction."""
    
    def __init__(self):
        self.cache = {}
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_comprehensive_stock_data(ticker: str, period: str = "1y") -> Dict:
        """
        Get comprehensive stock data including all available yfinance data.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
        
        Returns:
            Dict: Comprehensive stock data
        """
        try:
            stock = yf.Ticker(ticker)
            data = {}
            
            # Historical price data
            data['history'] = stock.history(period=period)
            
            # Company info
            try:
                data['info'] = stock.info
            except Exception as e:
                logger.warning(f"Could not fetch info for {ticker}: {e}")
                data['info'] = {}
            
            # Financial statements
            data['financials'] = {
                'income_stmt': stock.financials,
                'quarterly_income_stmt': stock.quarterly_financials,
                'balance_sheet': stock.balance_sheet,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_cash_flow': stock.quarterly_cashflow
            }
            
            # Analyst data
            try:
                data['recommendations'] = stock.recommendations
                data['analyst_price_targets'] = stock.analyst_price_targets
                data['revenue_forecasts'] = stock.revenue_forecasts
                data['earnings_forecasts'] = stock.earnings_forecasts
            except Exception as e:
                logger.warning(f"Could not fetch analyst data for {ticker}: {e}")
                data['recommendations'] = pd.DataFrame()
                data['analyst_price_targets'] = pd.DataFrame()
                data['revenue_forecasts'] = pd.DataFrame()
                data['earnings_forecasts'] = pd.DataFrame()
            
            # Dividend and splits data
            try:
                data['dividends'] = stock.dividends
                data['splits'] = stock.splits
            except Exception as e:
                logger.warning(f"Could not fetch dividend/splits data for {ticker}: {e}")
                data['dividends'] = pd.Series(dtype=float)
                data['splits'] = pd.Series(dtype=float)
            
            # Options data
            try:
                data['options_dates'] = stock.options
                if data['options_dates']:
                    # Get options chain for nearest expiration
                    nearest_exp = data['options_dates'][0]
                    option_chain = stock.option_chain(nearest_exp)
                    data['options_chain'] = {
                        'calls': option_chain.calls,
                        'puts': option_chain.puts,
                        'expiration': nearest_exp
                    }
                else:
                    data['options_chain'] = {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiration': None}
            except Exception as e:
                logger.warning(f"Could not fetch options data for {ticker}: {e}")
                data['options_dates'] = []
                data['options_chain'] = {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiration': None}
            
            # News data
            try:
                data['news'] = stock.news
            except Exception as e:
                logger.warning(f"Could not fetch news for {ticker}: {e}")
                data['news'] = []
            
            # ESG data
            try:
                data['sustainability'] = stock.sustainability
                data['esg_scores'] = stock.get_sustainability()
            except Exception as e:
                logger.warning(f"Could not fetch ESG data for {ticker}: {e}")
                data['sustainability'] = pd.DataFrame()
                data['esg_scores'] = None
            
            # Institutional holders
            try:
                data['institutional_holders'] = stock.institutional_holders
                data['major_holders'] = stock.major_holders
                data['mutualfund_holders'] = stock.mutualfund_holders
            except Exception as e:
                logger.warning(f"Could not fetch holders data for {ticker}: {e}")
                data['institutional_holders'] = pd.DataFrame()
                data['major_holders'] = pd.DataFrame()
                data['mutualfund_holders'] = pd.DataFrame()
            
            # Calendar events
            try:
                data['calendar'] = stock.calendar
                data['earnings_dates'] = stock.earnings_dates
            except Exception as e:
                logger.warning(f"Could not fetch calendar data for {ticker}: {e}")
                data['calendar'] = pd.DataFrame()
                data['earnings_dates'] = pd.DataFrame()
            
            # Market cap and shares data
            try:
                data['shares'] = {
                    'shares_outstanding': stock.get_shares_full(start="2022-01-01"),
                    'insider_purchases': stock.insider_purchases,
                    'insider_transactions': stock.insider_transactions,
                    'insider_roster_holders': stock.insider_roster_holders
                }
            except Exception as e:
                logger.warning(f"Could not fetch shares data for {ticker}: {e}")
                data['shares'] = {
                    'shares_outstanding': pd.DataFrame(),
                    'insider_purchases': pd.DataFrame(),
                    'insider_transactions': pd.DataFrame(),
                    'insider_roster_holders': pd.DataFrame()
                }
            
            logger.info(f"Successfully fetched comprehensive data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data for {ticker}: {e}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_multiple_stocks_data(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks efficiently.
        
        Args:
            tickers (List[str]): List of ticker symbols
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of ticker -> historical data
        """
        try:
            # Use yfinance's batch download feature
            data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, prepost=True)
            
            result = {}
            if len(tickers) == 1:
                result[tickers[0]] = data
            else:
                for ticker in tickers:
                    if ticker in data.columns.levels[0]:
                        result[ticker] = data[ticker].dropna()
                    else:
                        logger.warning(f"No data found for {ticker}")
                        result[ticker] = pd.DataFrame()
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching multiple stocks data: {e}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_market_indices() -> Dict[str, pd.DataFrame]:
        """Get data for major market indices."""
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        return EnhancedDataFetcher.get_multiple_stocks_data(list(indices.values()), "1y")
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_sector_etfs() -> Dict[str, str]:
        """Get popular sector ETFs for comparison."""
        return {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_crypto_data(symbols: List[str] = None, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get cryptocurrency data.
        
        Args:
            symbols (List[str]): Crypto symbols (default: popular ones)
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Crypto price data
        """
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD']
        
        return EnhancedDataFetcher.get_multiple_stocks_data(symbols, period)
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_currency_data(pairs: List[str] = None, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get currency pair data.
        
        Args:
            pairs (List[str]): Currency pairs (default: popular ones)
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Currency pair data
        """
        if pairs is None:
            pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
        
        return EnhancedDataFetcher.get_multiple_stocks_data(pairs, period)
    
    def calculate_advanced_metrics(self, data: pd.DataFrame, info: Dict) -> Dict:
        """
        Calculate advanced financial metrics.
        
        Args:
            data (pd.DataFrame): Historical price data
            info (Dict): Company info from yfinance
        
        Returns:
            Dict: Advanced metrics
        """
        metrics = {}
        
        if data.empty:
            return metrics
        
        try:
            # Price metrics
            current_price = data['Close'].iloc[-1]
            metrics['current_price'] = current_price
            metrics['52_week_high'] = data['Close'].rolling(window=252).max().iloc[-1]
            metrics['52_week_low'] = data['Close'].rolling(window=252).min().iloc[-1]
            metrics['52_week_change'] = (current_price / data['Close'].iloc[-252] - 1) * 100 if len(data) >= 252 else None
            
            # Volatility metrics
            returns = data['Close'].pct_change().dropna()
            metrics['volatility_30d'] = returns.tail(30).std() * np.sqrt(252) * 100
            metrics['volatility_90d'] = returns.tail(90).std() * np.sqrt(252) * 100
            metrics['volatility_1y'] = returns.std() * np.sqrt(252) * 100
            
            # Volume metrics
            avg_volume_30d = data['Volume'].tail(30).mean()
            avg_volume_90d = data['Volume'].tail(90).mean()
            current_volume = data['Volume'].iloc[-1]
            metrics['avg_volume_30d'] = avg_volume_30d
            metrics['volume_ratio'] = current_volume / avg_volume_30d if avg_volume_30d > 0 else 0
            
            # Performance metrics
            metrics['1d_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100 if len(data) >= 2 else 0
            metrics['5d_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100 if len(data) >= 6 else 0
            metrics['30d_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[-31] - 1) * 100 if len(data) >= 31 else 0
            metrics['90d_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[-91] - 1) * 100 if len(data) >= 91 else 0
            metrics['ytd_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            
            # From company info
            if info:
                metrics['market_cap'] = info.get('marketCap', 0)
                metrics['pe_ratio'] = info.get('trailingPE', 0)
                metrics['forward_pe'] = info.get('forwardPE', 0)
                metrics['pb_ratio'] = info.get('priceToBook', 0)
                metrics['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                metrics['beta'] = info.get('beta', 0)
                metrics['roe'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                metrics['roa'] = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
                metrics['debt_to_equity'] = info.get('debtToEquity', 0)
                metrics['current_ratio'] = info.get('currentRatio', 0)
                metrics['quick_ratio'] = info.get('quickRatio', 0)
                metrics['gross_margin'] = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
                metrics['operating_margin'] = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
                metrics['profit_margin'] = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
                
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
        
        return metrics

def format_large_number(value: Union[int, float, None]) -> str:
    """Format large numbers with appropriate suffixes."""
    if not isinstance(value, (int, float)) or pd.isna(value) or value == 0:
        return 'N/A'
    
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: Union[int, float, None], decimals: int = 2) -> str:
    """Format percentage values."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return 'N/A'
    return f"{value:.{decimals}f}%"

def format_ratio(value: Union[int, float, None], decimals: int = 2) -> str:
    """Format ratio values."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return 'N/A'
    return f"{value:.{decimals}f}"

# Global instance
data_fetcher = EnhancedDataFetcher()