import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import json
import uuid
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPortfolioManager:
    """Advanced portfolio management with comprehensive analytics and risk metrics."""
    
    def __init__(self):
        self.initialize_portfolio()
    
    def initialize_portfolio(self):
        """Initialize portfolio in session state."""
        if 'advanced_portfolio' not in st.session_state:
            st.session_state.advanced_portfolio = {
                'holdings': {},
                'transactions': [],
                'watchlist': [],
                'benchmarks': ['^GSPC', '^IXIC'],
                'settings': {
                    'risk_free_rate': 0.02,  # 2% annual risk-free rate
                    'base_currency': 'USD',
                    'rebalance_threshold': 0.05  # 5% threshold for rebalancing alerts
                },
                'performance_history': {},
                'created_date': datetime.now().isoformat()
            }
    
    def add_transaction(self, ticker: str, transaction_type: str, shares: float, price: float, 
                       date: datetime = None, fees: float = 0.0, notes: str = "") -> bool:
        """
        Add a transaction to the portfolio.
        
        Args:
            ticker (str): Stock ticker
            transaction_type (str): 'buy', 'sell', 'dividend'
            shares (float): Number of shares
            price (float): Price per share
            date (datetime): Transaction date
            fees (float): Transaction fees
            notes (str): Additional notes
        
        Returns:
            bool: Success status
        """
        try:
            if date is None:
                date = datetime.now()
            
            transaction = {
                'id': str(uuid.uuid4()),
                'ticker': ticker.upper(),
                'type': transaction_type.lower(),
                'shares': shares,
                'price': price,
                'date': date.isoformat(),
                'fees': fees,
                'notes': notes,
                'total_value': shares * price + (fees if transaction_type.lower() == 'buy' else -fees)
            }
            
            st.session_state.advanced_portfolio['transactions'].append(transaction)
            self.update_holdings()
            logger.info(f"Added transaction: {transaction_type} {shares} shares of {ticker} at ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return False
    
    def update_holdings(self):
        """Update portfolio holdings based on transactions."""
        holdings = {}
        
        for transaction in st.session_state.advanced_portfolio['transactions']:
            ticker = transaction['ticker']
            
            if ticker not in holdings:
                holdings[ticker] = {
                    'shares': 0,
                    'total_cost': 0,
                    'total_fees': 0,
                    'first_purchase_date': None,
                    'transactions': []
                }
            
            shares = transaction['shares']
            price = transaction['price']
            fees = transaction['fees']
            trans_date = datetime.fromisoformat(transaction['date'])
            
            if transaction['type'] == 'buy':
                holdings[ticker]['shares'] += shares
                holdings[ticker]['total_cost'] += shares * price + fees
                holdings[ticker]['total_fees'] += fees
                
                if holdings[ticker]['first_purchase_date'] is None:
                    holdings[ticker]['first_purchase_date'] = trans_date
                
            elif transaction['type'] == 'sell':
                holdings[ticker]['shares'] -= shares
                # Reduce cost basis proportionally
                if holdings[ticker]['shares'] + shares > 0:  # Avoid division by zero
                    cost_per_share = holdings[ticker]['total_cost'] / (holdings[ticker]['shares'] + shares)
                    holdings[ticker]['total_cost'] -= shares * cost_per_share
                holdings[ticker]['total_fees'] += fees
            
            holdings[ticker]['transactions'].append(transaction)
            
            # Calculate average cost per share
            if holdings[ticker]['shares'] > 0:
                holdings[ticker]['avg_cost_per_share'] = holdings[ticker]['total_cost'] / holdings[ticker]['shares']
            else:
                holdings[ticker]['avg_cost_per_share'] = 0
        
        # Remove positions with zero shares
        holdings = {k: v for k, v in holdings.items() if v['shares'] > 0}
        
        st.session_state.advanced_portfolio['holdings'] = holdings
    
    def get_current_portfolio_value(self) -> Dict:
        """Get current portfolio value and metrics."""
        holdings = st.session_state.advanced_portfolio['holdings']
        
        if not holdings:
            return {
                'total_value': 0,
                'total_cost': 0,
                'total_gain_loss': 0,
                'total_gain_loss_pct': 0,
                'positions': []
            }
        
        tickers = list(holdings.keys())
        current_prices = {}
        
        # Fetch current prices
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                current_prices[ticker] = stock.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {'error': 'Could not fetch current prices'}
        
        positions = []
        total_value = 0
        total_cost = 0
        
        for ticker, holding in holdings.items():
            current_price = current_prices.get(ticker, 0)
            position_value = holding['shares'] * current_price
            position_cost = holding['total_cost']
            gain_loss = position_value - position_cost
            gain_loss_pct = (gain_loss / position_cost * 100) if position_cost > 0 else 0
            
            position = {
                'ticker': ticker,
                'shares': holding['shares'],
                'avg_cost': holding['avg_cost_per_share'],
                'current_price': current_price,
                'position_value': position_value,
                'position_cost': position_cost,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'weight': 0,  # Will calculate after total_value is known
                'fees': holding['total_fees']
            }
            
            positions.append(position)
            total_value += position_value
            total_cost += position_cost
        
        # Calculate weights
        for position in positions:
            position['weight'] = (position['position_value'] / total_value * 100) if total_value > 0 else 0
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct,
            'positions': positions
        }
    
    def calculate_portfolio_metrics(self, period: str = "1y") -> Dict:
        """Calculate comprehensive portfolio performance metrics."""
        holdings = st.session_state.advanced_portfolio['holdings']
        
        if not holdings:
            return {}
        
        tickers = list(holdings.keys())
        weights = []
        
        # Get current portfolio composition
        portfolio_data = self.get_current_portfolio_value()
        
        if 'error' in portfolio_data:
            return {'error': portfolio_data['error']}
        
        # Calculate weights
        for position in portfolio_data['positions']:
            weights.append(position['weight'] / 100)
        
        # Fetch historical data
        try:
            stock_data = yf.download(tickers, period=period, auto_adjust=True)['Close']
            
            if len(tickers) == 1:
                stock_data = pd.DataFrame({tickers[0]: stock_data})
            
            if stock_data.empty:
                return {'error': 'No historical data available'}
            
            # Calculate returns
            returns = stock_data.pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Fetch benchmark data (S&P 500)
            benchmark = yf.download('^GSPC', period=period, auto_adjust=True)['Close']
            benchmark_returns = benchmark.pct_change().dropna()
            
            # Risk-free rate (from settings)
            risk_free_rate = st.session_state.advanced_portfolio['settings']['risk_free_rate']
            daily_rf_rate = risk_free_rate / 252  # Convert annual to daily
            
            # Calculate metrics
            metrics = self._calculate_risk_metrics(portfolio_returns, benchmark_returns, daily_rf_rate)
            
            # Add portfolio-specific metrics
            metrics['current_value'] = portfolio_data['total_value']
            metrics['total_cost'] = portfolio_data['total_cost']
            metrics['total_return'] = portfolio_data['total_gain_loss']
            metrics['total_return_pct'] = portfolio_data['total_gain_loss_pct']
            metrics['number_of_positions'] = len(portfolio_data['positions'])
            
            # Diversification metrics
            metrics['concentration_risk'] = max([pos['weight'] for pos in portfolio_data['positions']])
            metrics['effective_number_of_stocks'] = 1 / sum([(w/100)**2 for w in [pos['weight'] for pos in portfolio_data['positions']]])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> Dict:
        """Calculate risk and performance metrics."""
        metrics = {}
        
        try:
            # Basic return metrics
            metrics['daily_return_mean'] = portfolio_returns.mean()
            metrics['daily_return_std'] = portfolio_returns.std()
            metrics['annualized_return'] = portfolio_returns.mean() * 252
            metrics['annualized_volatility'] = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['annualized_volatility']
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdown.min()
            metrics['current_drawdown'] = drawdown.iloc[-1]
            
            # Calmar Ratio
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            # Sortino Ratio (using downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
            
            # Value at Risk (VaR)
            metrics['var_95'] = np.percentile(portfolio_returns, 5)
            metrics['var_99'] = np.percentile(portfolio_returns, 1)
            
            # Conditional Value at Risk (CVaR)
            var_95 = metrics['var_95']
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Beta and Alpha (vs benchmark)
            if not benchmark_returns.empty and len(benchmark_returns) > 1:
                # Align the time series
                aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
                if not aligned_data.empty and len(aligned_data) > 1:
                    port_ret = aligned_data.iloc[:, 0]
                    bench_ret = aligned_data.iloc[:, 1]
                    
                    # Calculate beta
                    covariance = np.cov(port_ret, bench_ret)[0][1]
                    benchmark_variance = np.var(bench_ret)
                    metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
                    
                    # Calculate alpha
                    portfolio_mean = port_ret.mean() * 252
                    benchmark_mean = bench_ret.mean() * 252
                    metrics['alpha'] = portfolio_mean - (risk_free_rate + metrics['beta'] * (benchmark_mean - risk_free_rate))
                    
                    # Information Ratio
                    excess_returns = port_ret - bench_ret
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    metrics['information_ratio'] = (portfolio_mean - benchmark_mean) / tracking_error if tracking_error != 0 else 0
                    
                    # Correlation with benchmark
                    metrics['correlation_with_benchmark'] = np.corrcoef(port_ret, bench_ret)[0][1]
            
            # Skewness and Kurtosis
            metrics['skewness'] = stats.skew(portfolio_returns.dropna())
            metrics['kurtosis'] = stats.kurtosis(portfolio_returns.dropna())
            
            # Win/Loss ratio
            winning_days = len(portfolio_returns[portfolio_returns > 0])
            losing_days = len(portfolio_returns[portfolio_returns < 0])
            metrics['win_rate'] = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def get_sector_allocation(self) -> Dict:
        """Get portfolio allocation by sector."""
        holdings = st.session_state.advanced_portfolio['holdings']
        
        if not holdings:
            return {}
        
        sector_allocation = {}
        total_value = 0
        
        portfolio_data = self.get_current_portfolio_value()
        if 'error' in portfolio_data:
            return {'error': portfolio_data['error']}
        
        for position in portfolio_data['positions']:
            ticker = position['ticker']
            value = position['position_value']
            
            try:
                # Get sector information
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                
                sector_allocation[sector] += value
                total_value += value
                
            except Exception as e:
                logger.warning(f"Could not get sector info for {ticker}: {e}")
                if 'Unknown' not in sector_allocation:
                    sector_allocation['Unknown'] = 0
                sector_allocation['Unknown'] += value
                total_value += value
        
        # Convert to percentages
        for sector in sector_allocation:
            sector_allocation[sector] = (sector_allocation[sector] / total_value * 100) if total_value > 0 else 0
        
        return sector_allocation
    
    def generate_rebalancing_suggestions(self) -> List[Dict]:
        """Generate portfolio rebalancing suggestions."""
        holdings = st.session_state.advanced_portfolio['holdings']
        
        if not holdings:
            return []
        
        suggestions = []
        portfolio_data = self.get_current_portfolio_value()
        
        if 'error' in portfolio_data:
            return [{'error': portfolio_data['error']}]
        
        # Check for concentration risk
        threshold = st.session_state.advanced_portfolio['settings']['rebalance_threshold'] * 100
        
        for position in portfolio_data['positions']:
            if position['weight'] > 25:  # More than 25% in single stock
                suggestions.append({
                    'type': 'concentration_risk',
                    'ticker': position['ticker'],
                    'current_weight': position['weight'],
                    'suggestion': f"Consider reducing {position['ticker']} position (currently {position['weight']:.1f}% of portfolio)",
                    'priority': 'high' if position['weight'] > 40 else 'medium'
                })
        
        # Check for drift from target allocations (if set)
        # This would require user-defined target allocations
        
        # Check for tax loss harvesting opportunities
        for position in portfolio_data['positions']:
            if position['gain_loss'] < -1000:  # Loss greater than $1000
                suggestions.append({
                    'type': 'tax_loss_harvesting',
                    'ticker': position['ticker'],
                    'current_loss': position['gain_loss'],
                    'suggestion': f"Consider tax loss harvesting for {position['ticker']} (unrealized loss: ${position['gain_loss']:.2f})",
                    'priority': 'low'
                })
        
        return suggestions
    
    def export_portfolio_data(self, format_type: str = 'csv') -> str:
        """Export portfolio data to various formats."""
        portfolio_data = self.get_current_portfolio_value()
        
        if 'error' in portfolio_data:
            return f"Error: {portfolio_data['error']}"
        
        if format_type.lower() == 'csv':
            df = pd.DataFrame(portfolio_data['positions'])
            return df.to_csv(index=False)
        
        elif format_type.lower() == 'json':
            return json.dumps(st.session_state.advanced_portfolio, indent=2, default=str)
        
        return "Unsupported format"
    
    def import_portfolio_data(self, data: str, format_type: str = 'csv') -> bool:
        """Import portfolio data from various formats."""
        try:
            if format_type.lower() == 'csv':
                df = pd.read_csv(pd.StringIO(data))
                # Process CSV data and add transactions
                for _, row in df.iterrows():
                    self.add_transaction(
                        ticker=row['ticker'],
                        transaction_type='buy',
                        shares=row['shares'],
                        price=row['avg_cost'],
                        date=datetime.now()
                    )
                return True
            
            elif format_type.lower() == 'json':
                imported_data = json.loads(data)
                st.session_state.advanced_portfolio = imported_data
                return True
            
        except Exception as e:
            logger.error(f"Error importing portfolio data: {e}")
            return False
        
        return False
    
    def add_to_watchlist(self, ticker: str) -> bool:
        """Add ticker to watchlist."""
        try:
            ticker = ticker.upper()
            watchlist = st.session_state.advanced_portfolio['watchlist']
            
            if ticker not in watchlist:
                watchlist.append(ticker)
                logger.info(f"Added {ticker} to watchlist")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return False
    
    def remove_from_watchlist(self, ticker: str) -> bool:
        """Remove ticker from watchlist."""
        try:
            ticker = ticker.upper()
            watchlist = st.session_state.advanced_portfolio['watchlist']
            
            if ticker in watchlist:
                watchlist.remove(ticker)
                logger.info(f"Removed {ticker} from watchlist")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
            return False
    
    def get_watchlist_data(self) -> Dict:
        """Get current data for watchlist stocks."""
        watchlist = st.session_state.advanced_portfolio['watchlist']
        
        if not watchlist:
            return {}
        
        watchlist_data = {}
        
        try:
            for ticker in watchlist:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                info = stock.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                    
                    watchlist_data[ticker] = {
                        'current_price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'sector': info.get('sector', 'Unknown')
                    }
        
        except Exception as e:
            logger.error(f"Error getting watchlist data: {e}")
        
        return watchlist_data

# Global instance
portfolio_manager = AdvancedPortfolioManager()