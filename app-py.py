import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import sys

# Import modules
from data_fetcher import get_stock_data, get_company_info, get_esg_data, get_competitors
from data_fetcher import get_news, get_analyst_estimates, get_income_statement, get_balance_sheet, get_cash_flow
from technical_analysis import apply_technical_indicators
from visualizations import display_stock_chart, display_returns_chart, display_rsi_chart, display_macd_chart
from visualizations import display_analyst_recommendations, display_esg_data, create_comparison_chart
from sentiment_analysis import get_sentiment_score, display_news
from financial_analysis import display_company_info, display_income_statement, display_balance_sheet, display_cash_flow
from recommendation_engine import generate_recommendation, display_recommendation_visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TICKER = "NVDA"
DEFAULT_PERIOD = "1Y"
PERIODS = {
    "1W": {"days": 7, "name": "1 Week"},
    "1M": {"days": 30, "name": "1 Month"},
    "3M": {"days": 90, "name": "3 Months"},
    "6M": {"days": 180, "name": "6 Months"},
    "1Y": {"days": 365, "name": "1 Year"},
    "2Y": {"days": 730, "name": "2 Years"},
    "5Y": {"days": 1825, "name": "5 Years"}
}
TECH_INDICATORS = ["RSI", "MACD", "Bollinger Bands"]
DEFAULT_INDICATORS = ["RSI", "MACD"]

def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Enhanced Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
def sidebar_inputs():
    """Gather user inputs from sidebar"""
    st.sidebar.title("Stock Analysis Dashboard")
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=100)
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker", value=DEFAULT_TICKER).upper()
    
    # Time period selection with more intuitive display
    period = st.sidebar.selectbox(
        "Select Time Period",
        options=list(PERIODS.keys()),
        format_func=lambda x: PERIODS[x]["name"],
        index=list(PERIODS.keys()).index(DEFAULT_PERIOD)
    )
    
    # Technical indicators
    st.sidebar.subheader("Technical Indicators")
    selected_indicators = []
    for indicator in TECH_INDICATORS:
        if st.sidebar.checkbox(indicator, indicator in DEFAULT_INDICATORS):
            selected_indicators.append(indicator)
    
    # Compare with sector/industry
    comparison_option = st.sidebar.radio(
        "Comparison View",
        ["None", "Industry", "Sector", "S&P 500"],
        index=1
    )
    
    # Portfolio tracking toggle
    enable_portfolio = st.sidebar.checkbox("Enable Portfolio Tracking", value=False)
    
    # Settings 
    st.sidebar.subheader("Settings")
    show_news = st.sidebar.checkbox("Show News", value=True)
    show_fundamentals = st.sidebar.checkbox("Show Fundamentals", value=True)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard uses data from Yahoo Finance and Alpha Vantage. "
        "For educational purposes only."
    )
    
    return {
        "ticker": ticker,
        "period": period,
        "period_days": PERIODS[period]["days"],
        "indicators": selected_indicators,
        "comparison": comparison_option,
        "enable_portfolio": enable_portfolio,
        "show_news": show_news,
        "show_fundamentals": show_fundamentals
    }

def display_header(ticker, company_info):
    """Display the dashboard header with key information"""
    if company_info and 'name' in company_info:
        company_name = company_info['name']
        st.title(f"{ticker} - {company_name}")
    else:
        st.title(f"{ticker} - Stock Analysis Dashboard")
    
    # Display current date and data timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Default to 1 year
    st.caption(f"Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Add refresh button
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()

def display_metrics(stock_data, sentiment_score, company_info, recommendation):
    """Display key metrics at the top of the dashboard"""
    if stock_data is None or stock_data.empty:
        st.warning("No stock data available to display metrics.")
        return
    
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Format daily return for display
    daily_return = stock_data['Close'].pct_change().iloc[-1] if 'Close' in stock_data.columns else 0
    daily_return_str = f"{daily_return:.2%}" if not pd.isna(daily_return) else "N/A"
    daily_return_color = "green" if daily_return > 0 else "red" if daily_return < 0 else "gray"
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            "Current Price", 
            f"${stock_data['Close'].iloc[-1]:.2f}" if 'Close' in stock_data.columns else "N/A",
            daily_return_str
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if 'MA50' in stock_data.columns:
            st.metric("50-Day MA", f"${stock_data['MA50'].iloc[-1]:.2f}")
        else:
            st.metric("50-Day MA", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if 'MA200' in stock_data.columns:
            st.metric("200-Day MA", f"${stock_data['MA200'].iloc[-1]:.2f}")
        else:
            st.metric("200-Day MA", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if company_info and 'marketCap' in company_info and company_info['marketCap'] != 'N/A':
            from data_fetcher import format_large_number
            st.metric("Market Cap", format_large_number(company_info['marketCap']))
        else:
            st.metric("Market Cap", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col5:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if sentiment_score:
            sentiment_category = sentiment_score.get("category", "Neutral")
            sentiment_score_value = sentiment_score.get("score", 0)
            st.metric("Sentiment", sentiment_category, f"{sentiment_score_value:.2f}")
        else:
            st.metric("Sentiment", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col6:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if recommendation:
            st.metric("Recommendation", recommendation)
        else:
            st.metric("Recommendation", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    setup_page()
    inputs = sidebar_inputs()
    
    # Calculate date range based on selected period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=inputs["period_days"])
    
    # Display header and initialize loading state
    with st.spinner('Loading dashboard...'):
        # Fetch basic company info first for header
        company_info = get_company_info(inputs["ticker"])
        display_header(inputs["ticker"], company_info)
        
        # Fetch all required data
        stock_data = get_stock_data(inputs["ticker"], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if stock_data is None or stock_data.empty:
            st.error(f"Unable to fetch data for {inputs['ticker']}. Please check the ticker symbol and try again.")
            return
            
        # Apply technical indicators
        stock_data = apply_technical_indicators(stock_data, inputs["indicators"])
        
        # Fetch additional data in parallel
        sentiment_score = get_sentiment_score(inputs["ticker"]) if inputs["show_news"] else None
        esg_data = get_esg_data(inputs["ticker"])
        
        # Generate recommendation
        if company_info and stock_data is not None:
            analyst_estimates = get_analyst_estimates(inputs["ticker"])
            analyst_consensus = compute_analyst_consensus(analyst_estimates) if analyst_estimates is not None else None
            recommendation, factors = generate_recommendation(inputs["ticker"], company_info, esg_data, sentiment_score, stock_data, analyst_consensus)
        else:
            recommendation, factors = "N/A", {}
        
        # Display top metrics
        display_metrics(stock_data, sentiment_score, company_info, recommendation)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Price Analysis", 
            "🔍 Technical Indicators", 
            "🌿 ESG Analysis", 
            "ℹ️ Company Info", 
            "📰 News & Sentiment", 
            "📊 Financials"
        ])
        
        with tab1:
            st.header("Stock Price Analysis")
            display_stock_chart(stock_data, inputs["ticker"], inputs["indicators"])
            
            # Add comparison chart if selected
            if inputs["comparison"] != "None":
                st.subheader(f"{inputs['comparison']} Comparison")
                with st.spinner('Fetching comparison data...'):
                    competitors = get_competitors(inputs["ticker"]) if inputs["comparison"] == "Industry" else []
                    comparison_data = compare_performance(inputs["ticker"], competitors, inputs["period_days"])
                    if comparison_data is not None:
                        comparison_chart = create_comparison_chart(
                            comparison_data, 
                            f"{inputs['comparison']} Performance Comparison - {inputs['period']}"
                        )
                        st.plotly_chart(comparison_chart, use_container_width=True)
                    else:
                        st.warning(f"No {inputs['comparison'].lower()} comparison data available.")
        
        with tab2:
            st.header("Technical Indicators")
            
            # Display selected technical indicators
            if "RSI" in inputs["indicators"]:
                display_rsi_chart(stock_data)
            
            if "MACD" in inputs["indicators"]:
                display_macd_chart(stock_data)
            
            if "Bollinger Bands" in inputs["indicators"]:
                st.subheader("Bollinger Bands (Included in Price Chart)")
                st.write("""
                Bollinger Bands are displayed directly on the price chart. They consist of:
                - A middle band (20-day simple moving average)
                - An upper band (middle band + 2 standard deviations)
                - A lower band (middle band - 2 standard deviations)
                
                These bands help identify volatility and potential overbought/oversold conditions.
                """)
            
            # Trading signals based on indicators
            st.subheader("Trading Signals")
            signals = generate_trading_signals(stock_data, inputs["indicators"])
            for signal, details in signals.items():
                st.markdown(f"**{signal}**: {details['description']}")
                st.markdown(f"**Signal**: {details['signal']}")
        
        with tab3:
            st.header("ESG Analysis")
            if esg_data is not None:
                display_esg_data(esg_data)
            else:
                st.warning("ESG data not available for this stock.")
                
            # Add ESG comparison with industry peers
            st.subheader("ESG Comparison")
            st.write("Coming soon: Compare this company's ESG scores with industry peers.")
        
        with tab4:
            st.header("Company Information")
            if company_info:
                display_company_info(company_info)
                
                # Show recommendation details
                st.subheader("Stock Recommendation")
                st.write(f"**Recommendation**: {recommendation}")
                st.write("**Based on the following factors:**")
                display_recommendation_visualization(recommendation, factors)
                
                # Show analyst recommendations
                st.subheader("Analyst Recommendations")
                analyst_estimates = get_analyst_estimates(inputs["ticker"])
                analyst_consensus = compute_analyst_consensus(analyst_estimates) if analyst_estimates is not None else None
                display_analyst_recommendations(analyst_consensus)
            else:
                st.warning("Company information not available.")
        
        with tab5:
            st.header("News & Sentiment Analysis")
            
            if inputs["show_news"]:
                news = get_news(inputs["ticker"])
                
                # Display sentiment overview
                col1, col2 = st.columns([1, 2])
                with col1:
                    if sentiment_score:
                        st.subheader("Sentiment Summary")
                        
                        # Create a gauge chart for sentiment
                        score = sentiment_score.get("score", 0)
                        category = sentiment_score.get("category", "Neutral")
                        st.write(f"Overall sentiment: **{category}**")
                        st.write(f"Sentiment score: **{score:.2f}**")
                        st.write(f"Subjectivity: **{sentiment_score.get('subjectivity', 0):.2f}**")
                        
                        # Display sentiment trend
                        st.markdown("### Sentiment Impact")
                        if category == "Positive":
                            st.success("The positive sentiment may contribute to short-term price increases.")
                        elif category == "Negative":
                            st.error("The negative sentiment may contribute to short-term price decreases.")
                        else:
                            st.info("The neutral sentiment suggests market indecision.")
                
                with col2:
                    display_news(news)
            else:
                st.info("News display is disabled. Enable it in the sidebar settings.")
        
        with tab6:
            st.header("Financial Statements")
            if inputs["show_fundamentals"]:
                # Create tabs for different financial statements
                fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
                
                with fin_tab1:
                    income_statement = get_income_statement(inputs["ticker"])
                    if income_statement is not None:
                        display_income_statement(income_statement)
                    else:
                        st.warning("Income statement data not available.")
                
                with fin_tab2:
                    balance_sheet = get_balance_sheet(inputs["ticker"])
                    if balance_sheet is not None:
                        display_balance_sheet(balance_sheet)
                    else:
                        st.warning("Balance sheet data not available.")
                
                with fin_tab3:
                    cash_flow = get_cash_flow(inputs["ticker"])
                    if cash_flow is not None:
                        display_cash_flow(cash_flow)
                    else:
                        st.warning("Cash flow data not available.")
                
                # Add financial ratio analysis
                st.subheader("Key Financial Ratios")
                if company_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P/E Ratio", f"{company_info.get('forwardPE', 'N/A'):.2f}" if isinstance(company_info.get('forwardPE'), (int, float)) else 'N/A')
                    with col2:
                        st.metric("Dividend Yield", f"{company_info.get('dividendYield', 'N/A'):.2%}" if isinstance(company_info.get('dividendYield'), (int, float)) else 'N/A')
                    with col3:
                        st.metric("Price to Book", f"{company_info.get('priceToBook', 'N/A'):.2f}" if isinstance(company_info.get('priceToBook'), (int, float)) else 'N/A')
            else:
                st.info("Financial analysis is disabled. Enable it in the sidebar settings.")
    
    # Footer
    st.markdown("---")
    st.caption("Data provided by Yahoo Finance and Alpha Vantage. This dashboard is for informational purposes only.")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
