import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import yfinance as yf
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import enhanced modules with fallback
try:
    from enhanced_data_fetcher import data_fetcher, format_large_number, format_percentage, format_ratio
    from advanced_technical_analysis import technical_analyzer
    from advanced_portfolio_manager import portfolio_manager
    from enhanced_visualizations import enhanced_viz
    from advanced_financial_analysis import financial_analyzer
    ENHANCED_MODULES_AVAILABLE = True
    logger.info("Enhanced modules loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced modules not available, using basic functionality: {e}")
    ENHANCED_MODULES_AVAILABLE = False

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
    
    /* Content blocks - revert to original transparent styling */
    .block-container {
        background: transparent;
        padding: 2rem;
        margin: 1rem;
    }
    
    /* Text colors - revert to default for main content */
    .stMarkdown, .stText, p, div, span, label {
        color: inherit;
    }
    
    /* Headers - revert to default */
    h1, h2, h3, h4, h5, h6 {
        color: inherit;
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
    
    /* Sidebar text styling - keep original background, just fix text colors */
    [data-testid="stSidebarContent"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebarContent"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebarContent"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebarContent"] .stTextInput label,
    [data-testid="stSidebarContent"] .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }
    
    /* Success alert in sidebar - keep styling minimal, just fix text */
    [data-testid="stSidebarContent"] [data-testid="stAlertContentSuccess"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    }
    
    /* Info alert in sidebar - keep styling minimal, just fix text */
    [data-testid="stSidebarContent"] [data-testid="stAlertContentInfo"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebarContent"] h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
    }
    
    /* Streamlit components */
    .stSelectbox label, .stTextInput label, .stCheckbox label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Metrics styling - revert to default */
    .stMetric label {
        color: inherit;
        font-weight: 500;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Basic data fetching functions for fallback
@st.cache_data(ttl=3600)
def basic_get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Basic stock data fetching using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            logger.warning(f"No data available for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def basic_get_company_info(ticker: str) -> Dict:
    """Basic company info fetching using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        return {}

def basic_format_large_number(value) -> str:
    """Basic number formatting."""
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

class StockDashboard:
    """Stock Analysis Dashboard - Enhanced or Basic version."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'current_ticker' not in st.session_state:
            st.session_state.current_ticker = 'AAPL'
        
        if 'period' not in st.session_state:
            st.session_state.period = '1y'
        
        # Initialize enhanced dashboard session state variables
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = {
                'RSI': True,
                'MACD': True,
                'Bollinger Bands': False,
                'Moving Averages': True,
                'Stochastic': False,
                'Volume': True
            }
        
        if 'comparison_stocks' not in st.session_state:
            st.session_state.comparison_stocks = []
        
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = {}
        
        if 'portfolio_transactions' not in st.session_state:
            st.session_state.portfolio_transactions = []
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.title("🚀 Stock Analysis Dashboard")
        
        # Stock selection
        ticker_input = st.sidebar.text_input(
            "Enter Stock Ticker",
            value=st.session_state.current_ticker,
            help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        if ticker_input != st.session_state.current_ticker:
            st.session_state.current_ticker = ticker_input
        
        # Time period selection
        period_options = {
            '1D': '1d', '5D': '5d', '1M': '1mo', '3M': '3mo',
            '6M': '6mo', '1Y': '1y', '2Y': '2y', '5Y': '5y'
        }
        
        selected_period = st.sidebar.selectbox(
            "Time Period",
            options=list(period_options.keys()),
            index=5,  # Default to 1Y
            help="Select the time period for analysis"
        )
        
        st.session_state.period = period_options[selected_period]
        
        # Display enhanced features availability
        if ENHANCED_MODULES_AVAILABLE:
            st.sidebar.success("✅ Enhanced features available!")
            st.sidebar.info("20+ technical indicators, portfolio management, and financial analysis")
        else:
            st.sidebar.warning("⚠️ Basic mode - Enhanced features unavailable")
            st.sidebar.info("Install enhanced modules for full functionality")
    
    def render_basic_overview(self):
        """Render basic stock overview when enhanced modules aren't available."""
        st.header(f"📈 {st.session_state.current_ticker} - Basic Stock Overview")
        
        # Fetch basic data
        with st.spinner("Loading stock data..."):
            stock_data = basic_get_stock_data(st.session_state.current_ticker, st.session_state.period)
            company_info = basic_get_company_info(st.session_state.current_ticker)
        
        if stock_data.empty:
            st.error("No data available for the selected stock.")
            return
        
        # Basic metrics
        if not stock_data.empty:
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price * 100) if prev_price != 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Change", f"${change:+.2f}", f"{change_pct:+.2f}%")
            with col3:
                st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
            with col4:
                market_cap = company_info.get('marketCap', 0)
                st.metric("Market Cap", basic_format_large_number(market_cap))
        
        # Basic chart
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f"{st.session_state.current_ticker} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Company information
        if company_info:
            st.subheader("Company Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Sector**: {company_info.get('sector', 'N/A')}")
                st.info(f"**Industry**: {company_info.get('industry', 'N/A')}")
                st.info(f"**Country**: {company_info.get('country', 'N/A')}")
            
            with col2:
                st.info(f"**P/E Ratio**: {company_info.get('trailingPE', 'N/A')}")
                st.info(f"**Dividend Yield**: {company_info.get('dividendYield', 0) * 100:.2f}%" if company_info.get('dividendYield') else "N/A")
                st.info(f"**Beta**: {company_info.get('beta', 'N/A')}")
    
    def render_enhanced_dashboard(self):
        """Render the full enhanced dashboard when modules are available."""
        # Import the enhanced dashboard class
        from enhanced_app import EnhancedStockDashboard
        
        enhanced_dashboard = EnhancedStockDashboard()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🔍 Technical Analysis", 
            "💰 Financial Analysis", 
            "💼 Portfolio", 
            "⚖️ Comparison"
        ])
        
        with tab1:
            enhanced_dashboard.render_overview_tab()
        
        with tab2:
            enhanced_dashboard.render_technical_analysis_tab()
        
        with tab3:
            enhanced_dashboard.render_financial_analysis_tab()
        
        with tab4:
            enhanced_dashboard.render_portfolio_tab()
        
        with tab5:
            enhanced_dashboard.render_comparison_tab()
    
    def run(self):
        """Main application runner."""
        # Render sidebar
        self.render_sidebar()
        
        # Render appropriate dashboard based on module availability
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.render_enhanced_dashboard()
            except Exception as e:
                st.error(f"Error loading enhanced dashboard: {e}")
                st.info("Falling back to basic mode...")
                self.render_basic_overview()
        else:
            self.render_basic_overview()
        
        # Footer
        st.markdown("---")
        if ENHANCED_MODULES_AVAILABLE:
            st.markdown(
                "<div style='text-align: center; color: #666;'>"
                "🚀 Enhanced Stock Analysis Dashboard | "
                "Powered by yfinance | "
                f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='text-align: center; color: #666;'>"
                "📈 Basic Stock Analysis Dashboard | "
                "Powered by yfinance | "
                "For enhanced features, ensure all modules are available"
                "</div>",
                unsafe_allow_html=True
            )

# Run the application
if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run()