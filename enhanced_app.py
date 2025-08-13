import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import yfinance as yf
from typing import Dict, List, Optional

# Import enhanced modules
from enhanced_data_fetcher import data_fetcher, format_large_number, format_percentage, format_ratio
from advanced_technical_analysis import technical_analyzer
from advanced_portfolio_manager import portfolio_manager
from enhanced_visualizations import enhanced_viz
from advanced_financial_analysis import financial_analyzer

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Enhanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Main content container */
    .stApp {
        background: transparent;
    }
    
    /* Content blocks */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    
    /* Text colors */
    .stMarkdown, .stText, p, div, span, label {
        color: #2c3e50 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-card h2, .metric-card h4 {
        color: white !important;
        margin: 0;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
    }
    
    .danger-metric {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white !important;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
    
    /* Streamlit components */
    .stSelectbox label, .stTextInput label, .stCheckbox label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Metrics styling */
    .stMetric label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedStockDashboard:
    """Enhanced Stock Analysis Dashboard with comprehensive features."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'current_ticker' not in st.session_state:
            st.session_state.current_ticker = 'AAPL'
        
        if 'period' not in st.session_state:
            st.session_state.period = '1y'
        
        if 'comparison_stocks' not in st.session_state:
            st.session_state.comparison_stocks = []
        
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = {
                'RSI': True,
                'MACD': True,
                'Bollinger Bands': False,
                'Moving Averages': True,
                'Stochastic': False,
                'Volume': True
            }
        
        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = {}
        
        if 'portfolio_transactions' not in st.session_state:
            st.session_state.portfolio_transactions = []
    
    def render_sidebar(self):
        """Render the sidebar with controls and navigation."""
        st.sidebar.title("🚀 Enhanced Stock Dashboard")
        
        # Stock selection
        ticker_input = st.sidebar.text_input(
            "Enter Stock Ticker",
            value=st.session_state.current_ticker,
            help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        if ticker_input != st.session_state.current_ticker:
            st.session_state.current_ticker = ticker_input
            st.rerun()
        
        # Time period selection
        period_options = {
            '1D': '1d', '5D': '5d', '1M': '1mo', '3M': '3mo',
            '6M': '6mo', '1Y': '1y', '2Y': '2y', '5Y': '5y', '10Y': '10y'
        }
        
        selected_period = st.sidebar.selectbox(
            "Time Period",
            options=list(period_options.keys()),
            index=5,  # Default to 1Y
            help="Select the time period for analysis"
        )
        
        st.session_state.period = period_options[selected_period]
        
        # Technical indicators selection
        st.sidebar.subheader("Technical Indicators")
        
        technical_options = {
            'Moving Averages': st.sidebar.checkbox('Moving Averages', value=True),
            'Bollinger Bands': st.sidebar.checkbox('Bollinger Bands', value=True),
            'RSI': st.sidebar.checkbox('RSI', value=True),
            'MACD': st.sidebar.checkbox('MACD', value=True),
            'Stochastic': st.sidebar.checkbox('Stochastic', value=False),
            'ADX': st.sidebar.checkbox('ADX', value=False),
            'Ichimoku': st.sidebar.checkbox('Ichimoku Cloud', value=False),
        }
        
        st.session_state.selected_indicators = technical_options
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📊 Analyze", help="Analyze current stock"):
                st.session_state.trigger_analysis = True
        
        with col2:
            if st.button("➕ Add to Portfolio", help="Add to portfolio"):
                st.session_state.show_add_transaction = True
        
        # Watchlist management
        st.sidebar.subheader("📋 Watchlist")
        
        for ticker in st.session_state.watchlist:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(ticker, key=f"watchlist_{ticker}"):
                    st.session_state.current_ticker = ticker
                    st.rerun()
            with col2:
                if st.button("✕", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                    st.session_state.watchlist.remove(ticker)
                    st.rerun()
        
        # Add to watchlist
        new_watchlist_ticker = st.sidebar.text_input("Add to Watchlist").upper()
        if st.sidebar.button("Add") and new_watchlist_ticker:
            if new_watchlist_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_watchlist_ticker)
                st.rerun()
    
    def render_overview_tab(self):
        """Render the overview tab with key metrics and charts."""
        st.header(f"📈 {st.session_state.current_ticker} - Stock Overview")
        
        # Fetch comprehensive data
        with st.spinner("Loading comprehensive stock data..."):
            stock_data = data_fetcher.get_comprehensive_stock_data(
                st.session_state.current_ticker, 
                st.session_state.period
            )
        
        if 'error' in stock_data:
            st.error(f"Error loading data: {stock_data['error']}")
            return
        
        # Company information
        info = stock_data.get('info', {})
        
        if info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Current Price</h4>
                    <h2>${info.get('currentPrice', 0):.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                change_pct = info.get('regularMarketChangePercent', 0) * 100
                color_class = 'success-metric' if change_pct >= 0 else 'danger-metric'
                st.markdown(f"""
                <div class="metric-card {color_class}">
                    <h4>Change</h4>
                    <h2>{change_pct:+.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Market Cap</h4>
                    <h2>{format_large_number(info.get('marketCap', 0))}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>P/E Ratio</h4>
                    <h2>{info.get('trailingPE', 0):.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Price chart with technical indicators
        history_data = stock_data.get('history', pd.DataFrame())
        
        if not history_data.empty:
            # Calculate technical indicators
            enhanced_data = technical_analyzer.calculate_all_indicators(history_data)
            
            # Create candlestick chart
            indicators_to_show = {k: v for k, v in st.session_state.selected_indicators.items() if v}
            
            chart = enhanced_viz.create_candlestick_chart(
                enhanced_data,
                title="",  # Remove duplicate title since tab already has header
                show_volume=True,
                indicators=indicators_to_show
            )
            
            st.plotly_chart(chart, use_container_width=True)
            
            # Key metrics
            metrics = data_fetcher.calculate_advanced_metrics(enhanced_data, info)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("52W High", f"${metrics.get('52_week_high', 0):.2f}")
            with col2:
                st.metric("52W Low", f"${metrics.get('52_week_low', 0):.2f}")
            with col3:
                st.metric("Volatility (30D)", f"{metrics.get('volatility_30d', 0):.1f}%")
            with col4:
                st.metric("Beta", f"{metrics.get('beta', 0):.2f}")
            with col5:
                st.metric("Dividend Yield", f"{metrics.get('dividend_yield', 0):.2f}%")
    
    def render_technical_analysis_tab(self):
        """Render technical analysis tab."""
        st.header("🔍 Technical Analysis")
        
        # Fetch data
        stock_data = data_fetcher.get_comprehensive_stock_data(
            st.session_state.current_ticker, 
            st.session_state.period
        )
        
        history_data = stock_data.get('history', pd.DataFrame())
        
        if history_data.empty:
            st.warning("No historical data available for technical analysis.")
            return
        
        # Calculate all indicators
        enhanced_data = technical_analyzer.calculate_all_indicators(history_data)
        
        # Technical indicators dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Oscillators")
            
            if 'RSI' in enhanced_data.columns:
                current_rsi = enhanced_data['RSI'].iloc[-1]
                rsi_signal = "🟢 Oversold" if current_rsi < 30 else "🔴 Overbought" if current_rsi > 70 else "🟡 Neutral"
                st.metric("RSI", f"{current_rsi:.1f}", help=rsi_signal)
            
            if 'Stoch_K' in enhanced_data.columns:
                current_stoch = enhanced_data['Stoch_K'].iloc[-1]
                stoch_signal = "🟢 Oversold" if current_stoch < 20 else "🔴 Overbought" if current_stoch > 80 else "🟡 Neutral"
                st.metric("Stochastic %K", f"{current_stoch:.1f}", help=stoch_signal)
            
            if 'Williams_R' in enhanced_data.columns:
                current_wr = enhanced_data['Williams_R'].iloc[-1]
                st.metric("Williams %R", f"{current_wr:.1f}")
        
        with col2:
            st.subheader("Trend Indicators")
            
            if 'MACD' in enhanced_data.columns and 'MACD_Signal' in enhanced_data.columns:
                macd_diff = enhanced_data['MACD'].iloc[-1] - enhanced_data['MACD_Signal'].iloc[-1]
                macd_signal = "🟢 Bullish" if macd_diff > 0 else "🔴 Bearish"
                st.metric("MACD Signal", macd_signal)
            
            if 'ADX' in enhanced_data.columns:
                current_adx = enhanced_data['ADX'].iloc[-1]
                trend_strength = "🟢 Strong" if current_adx > 25 else "🟡 Weak"
                st.metric("ADX", f"{current_adx:.1f}", help=f"Trend: {trend_strength}")
        
        # Generate trading signals
        signals = technical_analyzer.get_trading_signals(enhanced_data)
        
        if signals:
            st.subheader("📊 Trading Signals")
            
            signal_cols = st.columns(3)
            
            with signal_cols[0]:
                st.write("**Buy Signals**")
                buy_signals = []
                if signals.get('RSI_Oversold', pd.Series()).iloc[-1] if not signals.get('RSI_Oversold', pd.Series()).empty else False:
                    buy_signals.append("RSI Oversold")
                if signals.get('MACD_Bullish', pd.Series()).iloc[-1] if not signals.get('MACD_Bullish', pd.Series()).empty else False:
                    buy_signals.append("MACD Bullish Cross")
                if signals.get('Golden_Cross', pd.Series()).iloc[-1] if not signals.get('Golden_Cross', pd.Series()).empty else False:
                    buy_signals.append("Golden Cross")
                
                if buy_signals:
                    for signal in buy_signals:
                        st.success(f"🟢 {signal}")
                else:
                    st.info("No buy signals")
            
            with signal_cols[1]:
                st.write("**Sell Signals**")
                sell_signals = []
                if signals.get('RSI_Overbought', pd.Series()).iloc[-1] if not signals.get('RSI_Overbought', pd.Series()).empty else False:
                    sell_signals.append("RSI Overbought")
                if signals.get('MACD_Bearish', pd.Series()).iloc[-1] if not signals.get('MACD_Bearish', pd.Series()).empty else False:
                    sell_signals.append("MACD Bearish Cross")
                if signals.get('Death_Cross', pd.Series()).iloc[-1] if not signals.get('Death_Cross', pd.Series()).empty else False:
                    sell_signals.append("Death Cross")
                
                if sell_signals:
                    for signal in sell_signals:
                        st.error(f"🔴 {signal}")
                else:
                    st.info("No sell signals")
            
            with signal_cols[2]:
                st.write("**Neutral Signals**")
                neutral_signals = []
                if signals.get('BB_Squeeze', pd.Series()).iloc[-1] if not signals.get('BB_Squeeze', pd.Series()).empty else False:
                    neutral_signals.append("Bollinger Band Squeeze")
                if signals.get('Weak_Trend', pd.Series()).iloc[-1] if not signals.get('Weak_Trend', pd.Series()).empty else False:
                    neutral_signals.append("Weak Trend")
                
                if neutral_signals:
                    for signal in neutral_signals:
                        st.warning(f"🟡 {signal}")
                else:
                    st.info("No neutral signals")
        
        # Pattern detection
        patterns = technical_analyzer.detect_patterns(history_data)
        
        if any(patterns.values()):
            st.subheader("📈 Pattern Detection")
            
            for pattern_type, pattern_data in patterns.items():
                if pattern_data:
                    st.write(f"**{pattern_type.replace('_', ' ').title()}**: {len(pattern_data)} detected")
    
    def render_financial_analysis_tab(self):
        """Render financial analysis tab."""
        st.header("💰 Financial Analysis")
        
        with st.spinner("Analyzing financial statements..."):
            analysis = financial_analyzer.analyze_financial_statements(st.session_state.current_ticker)
        
        if 'error' in analysis:
            st.error(f"Error in financial analysis: {analysis['error']}")
            return
        
        # Financial health score
        health_score = analysis.get('financial_health_score', 0)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            score_color = 'success-metric' if health_score >= 70 else 'warning-metric' if health_score >= 40 else 'danger-metric'
            st.markdown(f"""
            <div class="metric-card {score_color}">
                <h3>Financial Health Score</h3>
                <h1>{health_score:.0f}/100</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Key financial metrics
        st.subheader("📊 Key Financial Metrics")
        
        income_analysis = analysis.get('income_statement_analysis', {})
        balance_analysis = analysis.get('balance_sheet_analysis', {})
        cf_analysis = analysis.get('cash_flow_analysis', {})
        ratios = analysis.get('financial_ratios', {})
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            profit_margin = income_analysis.get('current_profit_margin', 0)
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            revenue_growth = income_analysis.get('revenue_growth_annual', 0)
            st.metric("Revenue Growth", f"{revenue_growth:+.1f}%")
        
        with metric_cols[1]:
            current_ratio = balance_analysis.get('current_ratio', 0)
            st.metric("Current Ratio", f"{current_ratio:.2f}")
            
            debt_to_equity = balance_analysis.get('debt_to_equity', 0)
            st.metric("Debt/Equity", f"{debt_to_equity:.2f}")
        
        with metric_cols[2]:
            operating_cf = cf_analysis.get('current_operating_cf', 0)
            st.metric("Operating CF", format_large_number(operating_cf))
            
            free_cf = cf_analysis.get('free_cash_flow', 0)
            st.metric("Free Cash Flow", format_large_number(free_cf))
        
        with metric_cols[3]:
            roe = ratios.get('return_on_equity', 0)
            st.metric("ROE", f"{roe:.1f}%")
            
            roa = ratios.get('return_on_assets', 0)
            st.metric("ROA", f"{roa:.1f}%")
        
        # Financial trends
        st.subheader("📈 Growth Analysis")
        
        growth_analysis = analysis.get('growth_analysis', {})
        
        if growth_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                revenue_growth_rates = growth_analysis.get('revenue_growth_rates', [])
                if revenue_growth_rates:
                    st.write("**Revenue Growth Trend**")
                    trend = growth_analysis.get('revenue_growth_trend', 'Unknown')
                    st.info(f"Trend: {trend}")
                    
                    # Display growth rates
                    growth_df = pd.DataFrame({
                        'Year': [f'Year {i+1}' for i in range(len(revenue_growth_rates))],
                        'Growth Rate (%)': revenue_growth_rates
                    })
                    st.bar_chart(growth_df.set_index('Year'))
            
            with col2:
                cagr = growth_analysis.get('revenue_cagr', 0)
                if cagr:
                    st.write("**Revenue CAGR**")
                    st.metric("Compound Annual Growth Rate", f"{cagr:.1f}%")
        
        # Assessments
        st.subheader("🔍 Financial Assessments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            liquidity_assessment = balance_analysis.get('liquidity_assessment', 'Unknown')
            st.info(f"**Liquidity**: {liquidity_assessment}")
        
        with col2:
            debt_assessment = balance_analysis.get('debt_assessment', 'Unknown')
            st.info(f"**Debt Level**: {debt_assessment}")
    
    def render_portfolio_tab(self):
        """Render portfolio management tab."""
        st.header("💼 Portfolio Management")
        
        # Portfolio overview
        portfolio_data = portfolio_manager.get_current_portfolio_value()
        
        if 'error' in portfolio_data:
            st.warning("No portfolio data available. Add some transactions to get started!")
        else:
            # Portfolio summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", format_large_number(portfolio_data['total_value']))
            with col2:
                st.metric("Total Cost", format_large_number(portfolio_data['total_cost']))
            with col3:
                gain_loss = portfolio_data['total_gain_loss']
                st.metric("Total Gain/Loss", 
                         format_large_number(gain_loss),
                         delta=f"{portfolio_data['total_gain_loss_pct']:+.1f}%")
            with col4:
                st.metric("Positions", len(portfolio_data['positions']))
            
            # Portfolio positions
            if portfolio_data['positions']:
                st.subheader("📊 Current Positions")
                
                positions_df = pd.DataFrame(portfolio_data['positions'])
                
                # Format the dataframe for display
                display_df = positions_df.copy()
                display_df['position_value'] = display_df['position_value'].apply(format_large_number)
                display_df['gain_loss'] = display_df['gain_loss'].apply(format_large_number)
                display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
                display_df['avg_cost'] = display_df['avg_cost'].apply(lambda x: f"${x:.2f}")
                display_df['gain_loss_pct'] = display_df['gain_loss_pct'].apply(lambda x: f"{x:+.1f}%")
                display_df['weight'] = display_df['weight'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Sector allocation
                sector_data = portfolio_manager.get_sector_allocation()
                if sector_data and 'error' not in sector_data:
                    st.subheader("🏭 Sector Allocation")
                    sector_chart = enhanced_viz.create_sector_allocation_pie(sector_data)
                    st.plotly_chart(sector_chart, use_container_width=True)
        
        # Add transaction form
        st.subheader("➕ Add Transaction")
        
        with st.form("add_transaction"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                transaction_ticker = st.text_input("Ticker", value=st.session_state.current_ticker)
            with col2:
                transaction_type = st.selectbox("Type", ["buy", "sell"])
            with col3:
                shares = st.number_input("Shares", min_value=0.01, step=0.01)
            with col4:
                price = st.number_input("Price per Share", min_value=0.01, step=0.01)
            
            col5, col6 = st.columns(2)
            with col5:
                fees = st.number_input("Fees", min_value=0.0, step=0.01)
            with col6:
                transaction_date = st.date_input("Date", value=datetime.now().date())
            
            notes = st.text_area("Notes (Optional)")
            
            if st.form_submit_button("Add Transaction"):
                success = portfolio_manager.add_transaction(
                    transaction_ticker.upper(),
                    transaction_type,
                    shares,
                    price,
                    datetime.combine(transaction_date, datetime.min.time()),
                    fees,
                    notes
                )
                
                if success:
                    st.success("Transaction added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add transaction")
        
        # Portfolio analytics
        if portfolio_data and 'error' not in portfolio_data and portfolio_data['positions']:
            st.subheader("📈 Portfolio Analytics")
            
            with st.spinner("Calculating portfolio metrics..."):
                metrics = portfolio_manager.calculate_portfolio_metrics(st.session_state.period)
            
            if metrics and 'error' not in metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Beta", f"{metrics.get('beta', 0):.2f}")
                
                with col2:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
                    st.metric("Volatility", f"{metrics.get('annualized_volatility', 0)*100:.1f}%")
                
                with col3:
                    st.metric("Alpha", f"{metrics.get('alpha', 0)*100:.2f}%")
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
    
    def render_comparison_tab(self):
        """Render stock comparison tab."""
        st.header("⚖️ Stock Comparison")
        
        # Stock selection for comparison
        col1, col2 = st.columns([3, 1])
        
        with col1:
            comparison_tickers = st.multiselect(
                "Select stocks to compare",
                options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ', 'IWM'],
                default=[st.session_state.current_ticker, 'SPY'] if st.session_state.current_ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ', 'IWM'] else ['AAPL', 'SPY'],
                help="Select 2-5 stocks for comparison"
            )
        
        with col2:
            if st.button("🔄 Compare"):
                if len(comparison_tickers) >= 2:
                    st.session_state.comparison_stocks = comparison_tickers
                    st.rerun()
        
        if st.session_state.comparison_stocks:
            # Fetch comparison data
            with st.spinner("Loading comparison data..."):
                comparison_data = data_fetcher.get_multiple_stocks_data(
                    st.session_state.comparison_stocks, 
                    st.session_state.period
                )
            
            if comparison_data:
                # Performance comparison chart
                st.subheader("📊 Price Performance Comparison")
                
                performance_chart = enhanced_viz.create_performance_comparison(
                    comparison_data,
                    "Stock Performance Comparison"
                )
                st.plotly_chart(performance_chart, use_container_width=True)
                
                # Metrics comparison
                st.subheader("📈 Key Metrics Comparison")
                
                comparison_metrics = []
                
                for ticker in st.session_state.comparison_stocks:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        metrics = {
                            'Ticker': ticker,
                            'Market Cap': format_large_number(info.get('marketCap', 0)),
                            'P/E Ratio': f"{info.get('trailingPE', 0):.2f}",
                            'P/B Ratio': f"{info.get('priceToBook', 0):.2f}",
                            'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "0%",
                            'Beta': f"{info.get('beta', 0):.2f}",
                            'ROE': f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else "N/A",
                            'Debt/Equity': f"{info.get('debtToEquity', 0):.2f}",
                            'Profit Margin': f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else "N/A"
                        }
                        
                        comparison_metrics.append(metrics)
                        
                    except Exception as e:
                        logger.error(f"Error fetching comparison data for {ticker}: {e}")
                
                if comparison_metrics:
                    comparison_df = pd.DataFrame(comparison_metrics)
                    st.dataframe(comparison_df, use_container_width=True)
    
    def run(self):
        """Main application runner."""
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🔍 Technical Analysis", 
            "💰 Financial Analysis", 
            "💼 Portfolio", 
            "⚖️ Comparison"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_technical_analysis_tab()
        
        with tab3:
            self.render_financial_analysis_tab()
        
        with tab4:
            self.render_portfolio_tab()
        
        with tab5:
            self.render_comparison_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🚀 Enhanced Stock Analysis Dashboard | "
            "Powered by yfinance | "
            f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            "</div>",
            unsafe_allow_html=True
        )

# Run the application
if __name__ == "__main__":
    dashboard = EnhancedStockDashboard()
    dashboard.run()