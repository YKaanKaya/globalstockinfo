import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import logging
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_portfolio():
    """
    Initialize portfolio in session state if it doesn't exist.
    """
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},
            'transactions': [],
            'watchlist': [],
            'performance': {
                'initial_value': 0,
                'current_value': 0,
                'daily_values': {}
            }
        }

def add_to_watchlist(ticker):
    """
    Add a stock to the watchlist.
    
    Args:
        ticker (str): Stock ticker symbol
    """
    try:
        initialize_portfolio()
        
        # Check if ticker is valid
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if 'symbol' not in info:
            st.error(f"Invalid ticker symbol: {ticker}")
            return False
        
        # Add to watchlist if not already there
        if ticker not in st.session_state.portfolio['watchlist']:
            st.session_state.portfolio['watchlist'].append(ticker)
            logger.info(f"Added {ticker} to watchlist")
            return True
        else:
            st.warning(f"{ticker} is already in your watchlist")
            return False
    except Exception as e:
        logger.error(f"Error adding {ticker} to watchlist: {str(e)}")
        st.error(f"Error adding {ticker} to watchlist: {str(e)}")
        return False

def remove_from_watchlist(ticker):
    """
    Remove a stock from the watchlist.
    
    Args:
        ticker (str): Stock ticker symbol
    """
    try:
        initialize_portfolio()
        
        if ticker in st.session_state.portfolio['watchlist']:
            st.session_state.portfolio['watchlist'].remove(ticker)
            logger.info(f"Removed {ticker} from watchlist")
            return True
        else:
            st.warning(f"{ticker} is not in your watchlist")
            return False
    except Exception as e:
        logger.error(f"Error removing {ticker} from watchlist: {str(e)}")
        st.error(f"Error removing {ticker} from watchlist: {str(e)}")
        return False

def add_transaction(ticker, transaction_type, shares, price, date=None):
    """
    Add a transaction to the portfolio.
    
    Args:
        ticker (str): Stock ticker symbol
        transaction_type (str): 'buy' or 'sell'
        shares (float): Number of shares
        price (float): Price per share
        date (str, optional): Transaction date in 'YYYY-MM-DD' format. Defaults to today.
    """
    try:
        initialize_portfolio()
        
        # Validate inputs
        if transaction_type not in ['buy', 'sell']:
            st.error("Transaction type must be 'buy' or 'sell'")
            return False
        
        if shares <= 0:
            st.error("Shares must be greater than 0")
            return False
        
        if price <= 0:
            st.error("Price must be greater than 0")
            return False
        
        # Set default date to today if not provided
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # For sell transactions, check if there are enough shares
        if transaction_type == 'sell':
            current_shares = 0
            if ticker in st.session_state.portfolio['holdings']:
                current_shares = st.session_state.portfolio['holdings'][ticker]['shares']
            
            if shares > current_shares:
                st.error(f"Cannot sell {shares} shares of {ticker}. You only have {current_shares} shares.")
                return False
        
        # Create transaction record
        transaction = {
            'id': str(uuid.uuid4()),
            'ticker': ticker,
            'type': transaction_type,
            'shares': shares,
            'price': price,
            'date': date,
            'value': shares * price
        }
        
        # Add to transactions list
        st.session_state.portfolio['transactions'].append(transaction)
        
        # Update holdings
        update_holdings_after_transaction(transaction)
        
        # Update portfolio performance
        update_portfolio_performance()
        
        logger.info(f"Added {transaction_type} transaction for {shares} shares of {ticker} at ${price}")
        return True
    except Exception as e:
        logger.error(f"Error adding transaction: {str(e)}")
        st.error(f"Error adding transaction: {str(e)}")
        return False

def update_holdings_after_transaction(transaction):
    """
    Update portfolio holdings after a transaction.
    
    Args:
        transaction (dict): Transaction details
    """
    try:
        ticker = transaction['ticker']
        shares = transaction['shares']
        price = transaction['price']
        transaction_type = transaction['type']
        
        # Initialize holding if it doesn't exist
        if ticker not in st.session_state.portfolio['holdings']:
            st.session_state.portfolio['holdings'][ticker] = {
                'shares': 0,
                'cost_basis': 0,
                'current_price': price,
                'value': 0,
                'gain_loss': 0,
                'gain_loss_percent': 0
            }
        
        holding = st.session_state.portfolio['holdings'][ticker]
        
        if transaction_type == 'buy':
            # Calculate new cost basis
            total_cost = holding['cost_basis'] * holding['shares']
            new_cost = price * shares
            new_shares = holding['shares'] + shares
            
            holding['shares'] = new_shares
            holding['cost_basis'] = (total_cost + new_cost) / new_shares if new_shares > 0 else 0
            
        elif transaction_type == 'sell':
            # Reduce shares
            holding['shares'] -= shares
            
            # If no shares left, calculate realized gain/loss and remove
            if holding['shares'] <= 0:
                st.session_state.portfolio['holdings'].pop(ticker)
            
        # Update current value
        update_current_prices()
        
    except Exception as e:
        logger.error(f"Error updating holdings: {str(e)}")
        st.error(f"Error updating holdings: {str(e)}")

def update_current_prices():
    """
    Update current prices for all holdings and calculate gain/loss.
    """
    try:
        initialize_portfolio()
        
        tickers = list(st.session_state.portfolio['holdings'].keys())
        
        if not tickers:
            return
        
        # Fetch current prices
        data = yf.download(tickers, period="1d")['Close']
        
        # Update each holding
        for ticker, holding in st.session_state.portfolio['holdings'].items():
            if isinstance(data, pd.Series):
                current_price = data.iloc[-1]
            else:
                current_price = data[ticker].iloc[-1] if ticker in data else holding['current_price']
            
            holding['current_price'] = current_price
            holding['value'] = holding['shares'] * current_price
            holding['gain_loss'] = holding['value'] - (holding['cost_basis'] * holding['shares'])
            holding['gain_loss_percent'] = (holding['gain_loss'] / (holding['cost_basis'] * holding['shares'])) * 100 if holding['cost_basis'] > 0 else 0
        
        logger.info(f"Updated prices for {len(tickers)} holdings")
        
    except Exception as e:
        logger.error(f"Error updating prices: {str(e)}")
        st.error(f"Error updating prices: {str(e)}")

def update_portfolio_performance():
    """
    Update portfolio performance metrics.
    """
    try:
        initialize_portfolio()
        
        # Calculate current portfolio value
        total_value = sum(holding['value'] for holding in st.session_state.portfolio['holdings'].values())
        
        # Update performance data
        st.session_state.portfolio['performance']['current_value'] = total_value
        
        # Add today's value to historical data
        today = datetime.now().strftime('%Y-%m-%d')
        st.session_state.portfolio['performance']['daily_values'][today] = total_value
        
        # Calculate initial value from transactions
        buy_transactions = [t for t in st.session_state.portfolio['transactions'] if t['type'] == 'buy']
        sell_transactions = [t for t in st.session_state.portfolio['transactions'] if t['type'] == 'sell']
        
        total_invested = sum(t['value'] for t in buy_transactions)
        total_withdrawn = sum(t['value'] for t in sell_transactions)
        
        st.session_state.portfolio['performance']['initial_value'] = total_invested - total_withdrawn
        
        logger.info(f"Updated portfolio performance. Current value: ${total_value:.2f}")
        
    except Exception as e:
        logger.error(f"Error updating portfolio performance: {str(e)}")
        st.error(f"Error updating portfolio performance: {str(e)}")

def calculate_portfolio_metrics():
    """
    Calculate advanced portfolio metrics.
    
    Returns:
        dict: Portfolio metrics
    """
    try:
        initialize_portfolio()
        
        metrics = {
            'total_value': 0,
            'total_gain_loss': 0,
            'total_gain_loss_percent': 0,
            'sector_allocation': {},
            'top_performers': [],
            'bottom_performers': []
        }
        
        # Get holdings
        holdings = st.session_state.portfolio['holdings']
        
        if not holdings:
            return metrics
        
        # Calculate basic metrics
        total_cost = sum(h['cost_basis'] * h['shares'] for h in holdings.values())
        total_value = sum(h['value'] for h in holdings.values())
        
        metrics['total_value'] = total_value
        metrics['total_gain_loss'] = total_value - total_cost
        metrics['total_gain_loss_percent'] = (metrics['total_gain_loss'] / total_cost) * 100 if total_cost > 0 else 0
        
        # Get sector information and performance data
        tickers = list(holdings.keys())
        
        # Calculate sector allocation
        try:
            sector_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                value = holdings[ticker]['value']
                
                if sector in sector_data:
                    sector_data[sector] += value
                else:
                    sector_data[sector] = value
            
            # Convert to percentages
            metrics['sector_allocation'] = {sector: (value / total_value) * 100 for sector, value in sector_data.items()}
        except Exception as e:
            logger.warning(f"Error calculating sector allocation: {str(e)}")
            metrics['sector_allocation'] = {'Unknown': 100}
        
        # Identify top and bottom performers
        performers = [{'ticker': ticker, 'gain_loss_percent': holding['gain_loss_percent']} 
                    for ticker, holding in holdings.items()]
        
        sorted_performers = sorted(performers, key=lambda x: x['gain_loss_percent'], reverse=True)
        
        metrics['top_performers'] = sorted_performers[:3]
        metrics['bottom_performers'] = sorted_performers[-3:] if len(sorted_performers) >= 3 else sorted_performers
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}

def save_portfolio():
    """
    Save portfolio data to session state and as a downloadable file.
    
    Returns:
        bytes: Portfolio data as JSON
    """
    try:
        initialize_portfolio()
        
        # Convert portfolio to JSON
        portfolio_json = json.dumps(st.session_state.portfolio, indent=2)
        
        return portfolio_json.encode('utf-8')
    except Exception as e:
        logger.error(f"Error saving portfolio: {str(e)}")
        st.error(f"Error saving portfolio: {str(e)}")
        return None

def load_portfolio(portfolio_data):
    """
    Load portfolio data from uploaded file.
    
    Args:
        portfolio_data (bytes): Portfolio data as JSON
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Decode JSON data
        portfolio_json = portfolio_data.decode('utf-8')
        portfolio = json.loads(portfolio_json)
        
        # Validate portfolio data
        required_keys = ['holdings', 'transactions', 'watchlist', 'performance']
        if not all(key in portfolio for key in required_keys):
            st.error("Invalid portfolio data format")
            return False
        
        # Update session state
        st.session_state.portfolio = portfolio
        
        # Update current prices
        update_current_prices()
        
        logger.info("Portfolio loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading portfolio: {str(e)}")
        st.error(f"Error loading portfolio: {str(e)}")
        return False

def display_portfolio_summary():
    """
    Display a summary of the portfolio.
    """
    try:
        initialize_portfolio()
        
        st.subheader("Portfolio Summary")
        
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics()
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Value", 
                f"${metrics['total_value']:.2f}", 
                f"{metrics['total_gain_loss_percent']:.2f}%" if metrics['total_gain_loss_percent'] != 0 else None
            )
        
        with col2:
            st.metric(
                "Total Gain/Loss", 
                f"${metrics['total_gain_loss']:.2f}"
            )
        
        with col3:
            # Calculate number of positions
            num_positions = len(st.session_state.portfolio['holdings'])
            st.metric("Number of Positions", num_positions)
        
        # Holdings table
        st.markdown("### Current Holdings")
        
        if not st.session_state.portfolio['holdings']:
            st.info("No holdings in portfolio")
        else:
            # Create DataFrame for holdings
            holdings_data = []
            for ticker, holding in st.session_state.portfolio['holdings'].items():
                holdings_data.append({
                    'Ticker': ticker,
                    'Shares': holding['shares'],
                    'Cost Basis': holding['cost_basis'],
                    'Current Price': holding['current_price'],
                    'Current Value': holding['value'],
                    'Gain/Loss': holding['gain_loss'],
                    'Gain/Loss %': holding['gain_loss_percent']
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            
            # Apply formatting
            formatted_df = holdings_df.copy()
            formatted_df['Cost Basis'] = formatted_df['Cost Basis'].map('${:.2f}'.format)
            formatted_df['Current Price'] = formatted_df['Current Price'].map('${:.2f}'.format)
            formatted_df['Current Value'] = formatted_df['Current Value'].map('${:.2f}'.format)
            formatted_df['Gain/Loss'] = formatted_df['Gain/Loss'].map('${:.2f}'.format)
            formatted_df['Gain/Loss %'] = formatted_df['Gain/Loss %'].map('{:.2f}%'.format)
            
            st.dataframe(formatted_df)
        
        # Portfolio allocation pie chart
        st.markdown("### Portfolio Allocation")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(st.session_state.portfolio['holdings'].keys()),
                values=[holding['value'] for holding in st.session_state.portfolio['holdings'].values()],
                hole=.3,
                textinfo='label+percent'
            )
        ])
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector allocation
        if metrics['sector_allocation']:
            st.markdown("### Sector Allocation")
            
            sector_fig = go.Figure(data=[
                go.Pie(
                    labels=list(metrics['sector_allocation'].keys()),
                    values=list(metrics['sector_allocation'].values()),
                    hole=.3,
                    textinfo='label+percent'
                )
            ])
            
            sector_fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(sector_fig, use_container_width=True)
        
        # Top and bottom performers
        if metrics['top_performers'] or metrics['bottom_performers']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top Performers")
                if metrics['top_performers']:
                    for p in metrics['top_performers']:
                        st.metric(
                            p['ticker'], 
                            f"{p['gain_loss_percent']:.2f}%"
                        )
                else:
                    st.info("No data available")
            
            with col2:
                st.markdown("### Bottom Performers")
                if metrics['bottom_performers']:
                    for p in metrics['bottom_performers']:
                        st.metric(
                            p['ticker'], 
                            f"{p['gain_loss_percent']:.2f}%"
                        )
                else:
                    st.info("No data available")
        
    except Exception as e:
        logger.error(f"Error displaying portfolio summary: {str(e)}")
        st.error(f"Error displaying portfolio summary: {str(e)}")

def display_transaction_history():
    """
    Display transaction history.
    """
    try:
        initialize_portfolio()
        
        st.subheader("Transaction History")
        
        transactions = st.session_state.portfolio['transactions']
        
        if not transactions:
            st.info("No transactions recorded")
            return
        
        # Sort transactions by date (newest first)
        sorted_transactions = sorted(transactions, key=lambda x: x['date'], reverse=True)
        
        # Create DataFrame
        trans_df = pd.DataFrame(sorted_transactions)
        
        # Format columns
        trans_df['type'] = trans_df['type'].str.capitalize()
        trans_df['date'] = pd.to_datetime(trans_df['date'])
        
        # Format for display
        display_df = trans_df[['date', 'ticker', 'type', 'shares', 'price', 'value']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['price'] = display_df['price'].map('${:.2f}'.format)
        display_df['value'] = display_df['value'].map('${:.2f}'.format)
        
        # Rename columns
        display_df.columns = ['Date', 'Ticker', 'Type', 'Shares', 'Price', 'Total Value']
        
        st.dataframe(display_df)
        
        # Add a chart of cumulative investment over time
        st.markdown("### Cumulative Investment Over Time")
        
        # Calculate cumulative investment
        trans_df = pd.DataFrame(sorted_transactions)
        trans_df['date'] = pd.to_datetime(trans_df['date'])
        trans_df = trans_df.sort_values('date')
        
        # Apply multiplier based on transaction type
        trans_df['adjusted_value'] = trans_df.apply(
            lambda row: row['value'] if row['type'] == 'buy' else -row['value'], 
            axis=1
        )
        
        # Calculate cumulative sum
        trans_df['cumulative_investment'] = trans_df['adjusted_value'].cumsum()
        
        # Create line chart
        fig = px.line(
            trans_df, 
            x='date', 
            y='cumulative_investment',
            markers=True,
            labels={'date': 'Date', 'cumulative_investment': 'Cumulative Investment ($)'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying transaction history: {str(e)}")
        st.error(f"Error displaying transaction history: {str(e)}")

def display_watchlist():
    """
    Display watchlist with current prices.
    """
    try:
        initialize_portfolio()
        
        st.subheader("Watchlist")
        
        watchlist = st.session_state.portfolio['watchlist']
        
        if not watchlist:
            st.info("Your watchlist is empty")
            return
        
        # Fetch current data for watchlist
        data = yf.download(watchlist, period="2d")
        
        if data.empty:
            st.warning("Could not fetch data for watchlist")
            return
        
        # Get latest prices and calculate daily change
        latest_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest_close
        
        price_change = latest_close - prev_close
        price_change_pct = price_change / prev_close * 100
        
        # Create watchlist data
        watchlist_data = []
        
        for ticker in watchlist:
            if isinstance(latest_close, pd.Series):
                current_price = latest_close[ticker] if ticker in latest_close else None
                daily_change = price_change[ticker] if ticker in price_change else None
                daily_change_pct = price_change_pct[ticker] if ticker in price_change_pct else None
            else:
                current_price = latest_close
                daily_change = price_change
                daily_change_pct = price_change_pct
            
            if current_price is not None:
                watchlist_data.append({
                    'Ticker': ticker,
                    'Current Price': current_price,
                    'Daily Change': daily_change,
                    'Daily Change %': daily_change_pct
                })
        
        if watchlist_data:
            watchlist_df = pd.DataFrame(watchlist_data)
            
            # Apply formatting
            formatted_df = watchlist_df.copy()
            formatted_df['Current Price'] = formatted_df['Current Price'].map('${:.2f}'.format)
            formatted_df['Daily Change'] = formatted_df['Daily Change'].map('${:.2f}'.format)
            formatted_df['Daily Change %'] = formatted_df['Daily Change %'].map('{:.2f}%'.format)
            
            st.dataframe(formatted_df)
            
            # Add action buttons
            ticker_to_remove = st.selectbox("Select ticker to remove from watchlist", watchlist)
            if st.button("Remove from Watchlist"):
                if remove_from_watchlist(ticker_to_remove):
                    st.success(f"Removed {ticker_to_remove} from watchlist")
                    st.experimental_rerun()
            
            ticker_to_buy = st.selectbox("Select ticker to add to portfolio", watchlist)
            col1, col2 = st.columns(2)
            with col1:
                shares = st.number_input("Number of shares", min_value=0.01, step=0.01)
            with col2:
                price = st.number_input("Price per share", min_value=0.01, step=0.01, 
                                       value=float(watchlist_df[watchlist_df['Ticker'] == ticker_to_buy]['Current Price'].iloc[0]))
            
            if st.button("Add to Portfolio"):
                if add_transaction(ticker_to_buy, "buy", shares, price):
                    st.success(f"Added {shares} shares of {ticker_to_buy} to portfolio")
        else:
            st.warning("Could not fetch data for any watchlist stocks")
        
    except Exception as e:
        logger.error(f"Error displaying watchlist: {str(e)}")
        st.error(f"Error displaying watchlist: {str(e)}")

def portfolio_management_ui():
    """
    Display portfolio management UI.
    """
    try:
        initialize_portfolio()
        
        st.title("Portfolio Management")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Portfolio Summary", 
            "Add Transaction", 
            "Transaction History", 
            "Watchlist"
        ])
        
        with tab1:
            display_portfolio_summary()
            
            # Add import/export functionality
            st.markdown("### Import/Export Portfolio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export portfolio
                if st.button("Export Portfolio"):
                    portfolio_data = save_portfolio()
                    if portfolio_data:
                        st.download_button(
                            label="Download Portfolio Data",
                            data=portfolio_data,
                            file_name="portfolio.json",
                            mime="application/json"
                        )
            
            with col2:
                # Import portfolio
                uploaded_file = st.file_uploader("Import Portfolio", type="json")
                if uploaded_file is not None:
                    portfolio_data = uploaded_file.read()
                    if load_portfolio(portfolio_data):
                        st.success("Portfolio loaded successfully")
                        st.experimental_rerun()
        
        with tab2:
            st.subheader("Add Transaction")
            
            # Transaction form
            ticker = st.text_input("Ticker Symbol").upper()
            
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_type = st.selectbox("Transaction Type", ["buy", "sell"])
                shares = st.number_input("Number of Shares", min_value=0.01, step=0.01)
            
            with col2:
                price = st.number_input("Price per Share", min_value=0.01, step=0.01)
                date = st.date_input("Transaction Date", datetime.now())
            
            if st.button("Add Transaction"):
                if ticker and shares > 0 and price > 0:
                    if add_transaction(ticker, transaction_type, shares, price, date.strftime('%Y-%m-%d')):
                        st.success(f"Added {transaction_type} transaction for {shares} shares of {ticker}")
                else:
                    st.error("Please fill in all fields")
            
            # Watchlist section within transactions tab
            st.subheader("Add to Watchlist")
            
            watchlist_ticker = st.text_input("Ticker Symbol for Watchlist").upper()
            
            if st.button("Add to Watchlist"):
                if watchlist_ticker:
                    if add_to_watchlist(watchlist_ticker):
                        st.success(f"Added {watchlist_ticker} to watchlist")
                else:
                    st.error("Please enter a ticker symbol")
        
        with tab3:
            display_transaction_history()
        
        with tab4:
            display_watchlist()
        
    except Exception as e:
        logger.error(f"Error in portfolio management UI: {str(e)}")
        st.error(f"Error in portfolio management: {str(e)}")

# Export function to be called from main app
def display_portfolio_tracker():
    """
    Main function to display portfolio tracker in the dashboard.
    """
    portfolio_management_ui()
