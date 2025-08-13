# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Enhanced GlobalStockInfo** is a professional-grade, comprehensive stock analysis dashboard built entirely with **yfinance** and **Streamlit**. The application has been completely redesigned with advanced technical analysis, sophisticated portfolio management, comprehensive financial analysis, and modern interactive visualizations. It provides institutional-quality analytics accessible through a user-friendly web interface.

## Enhanced Architecture (2025)

### Primary Application
- **enhanced_app.py** - Main enhanced application with modern UI and comprehensive features

### Core Enhanced Modules
- **enhanced_data_fetcher.py** - Comprehensive data acquisition using only yfinance (no external API keys required)
- **advanced_technical_analysis.py** - 20+ technical indicators, pattern detection, support/resistance analysis
- **advanced_portfolio_manager.py** - Professional portfolio analytics with risk metrics (Sharpe, Sortino, VaR, etc.)
- **enhanced_visualizations.py** - Modern interactive charts with Plotly (candlestick, heatmaps, risk-return plots)
- **advanced_financial_analysis.py** - Complete financial statement analysis and stock screening

### Legacy Files (Maintained for Reference)
- **app-py.py**, **app.py**, **globalstock.py** - Original implementations
- **data-fetcher-py.py**, **technical-analysis-py.py**, etc. - Original modules

## Dependencies and Setup

### Enhanced Dependencies
All dependencies are listed in `requirements.txt` (updated for 2025):
```bash
# Core - No external API keys needed
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.65  # Only data source required

# Visualization & Analysis
plotly>=5.17.0
matplotlib>=3.7.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Utilities
requests>=2.31.0
textblob>=0.17.1
python-dateutil>=2.8.2
pytz>=2023.3
openpyxl>=3.1.0  # For Excel export
xlsxwriter>=3.1.0
```

### Installation and Running
```bash
pip install -r requirements.txt
# Primary enhanced application
streamlit run enhanced_app.py

# Legacy applications (still functional)
streamlit run app-py.py
streamlit run globalstock.py
```

### No API Keys Required
The enhanced version uses **only yfinance**, eliminating the need for:
- Alpha Vantage API keys
- External API registrations
- Rate limit management beyond yfinance's built-in handling

## Enhanced Features Architecture

### 1. Advanced Technical Analysis
- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ADX, Ichimoku Cloud, Williams %R, CCI, ROC, Momentum, Parabolic SAR
- **Pattern Detection**: Double tops/bottoms, head and shoulders, triangles, flags, pennants
- **Support/Resistance Analysis**: Automated level identification with touch validation
- **Trading Signals**: Multi-indicator consensus system for buy/sell/neutral signals
- **Volume Analysis**: OBV, VWAP, A/D Line, Chaikin Money Flow

### 2. Professional Portfolio Management
- **Advanced Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, Maximum Drawdown, VaR (95%, 99%), CVaR
- **Performance Analytics**: Alpha, Beta, Information ratio, tracking error, correlation analysis
- **Portfolio Optimization**: Sector allocation, concentration risk analysis, rebalancing suggestions
- **Transaction Management**: Complete buy/sell history, cost basis tracking, tax loss harvesting identification
- **Benchmark Comparison**: S&P 500, NASDAQ, custom benchmarks

### 3. Comprehensive Financial Analysis
- **Financial Health Scoring**: Proprietary 0-100 scoring system based on profitability, liquidity, leverage, growth, and cash flow
- **Statement Analysis**: Automated analysis of income statements, balance sheets, and cash flow statements
- **Growth Analysis**: Revenue/earnings CAGR, trend analysis, quarterly momentum
- **Financial Ratios**: 15+ key ratios including ROE, ROA, current ratio, debt-to-equity, profit margins
- **Stock Screening**: Multi-criteria screening with customizable thresholds
- **Company Comparison**: Side-by-side financial metrics comparison

### 4. Enhanced Visualizations
- **Modern Candlestick Charts**: Interactive charts with technical indicator overlays
- **Risk-Return Analysis**: Scatter plots with efficient frontier visualization
- **Correlation Heatmaps**: Asset correlation analysis with color-coded matrices
- **Drawdown Analysis**: Portfolio drawdown visualization with recovery periods
- **Options Analysis**: Options chain visualization with implied volatility surfaces
- **Sector Allocation**: Interactive pie charts and allocation analysis

### 5. Market Coverage
- **Global Equities**: All stocks available through Yahoo Finance
- **Market Indices**: Major global indices (S&P 500, NASDAQ, international)
- **Sector Analysis**: Sector ETFs and industry group analysis
- **Alternative Assets**: Cryptocurrencies, forex pairs, commodities
- **Multi-Asset Portfolio**: Mixed asset class portfolio management

## Enhanced Data Flow

1. **User Input** → Ticker selection, timeframe, analysis preferences
2. **Comprehensive Data Fetch** → yfinance extracts all available data (OHLCV, financials, news, options, etc.)
3. **Advanced Processing** → Technical indicators, financial analysis, risk calculations
4. **Interactive Visualization** → Modern charts with real-time interactivity
5. **Portfolio Integration** → Transaction tracking, performance monitoring, risk management

## Development Patterns (Enhanced)

### Modern Code Architecture
- **Type Hints**: Full type annotation for better IDE support and debugging
- **Error Handling**: Comprehensive exception handling with user-friendly messages
- **Logging**: Structured logging for debugging and monitoring
- **Performance**: Streamlit caching, parallel processing, memory optimization
- **Modularity**: Clean separation of concerns across modules

### Session State Management
```python
# Enhanced session state for complex portfolio management
st.session_state.advanced_portfolio = {
    'holdings': {},          # Current positions
    'transactions': [],      # Complete transaction history
    'watchlist': [],         # User watchlist
    'benchmarks': [],        # Comparison benchmarks
    'settings': {},          # User preferences
    'performance_history': {}  # Historical performance tracking
}
```

### Caching Strategy
```python
# Optimized caching with appropriate TTL
@st.cache_data(ttl=3600)  # 1-hour cache for market data
def get_comprehensive_stock_data(ticker: str, period: str) -> Dict:
    # Comprehensive data extraction
```

## Testing and Quality Assurance

### Module Testing
```python
# Example comprehensive testing
from enhanced_data_fetcher import data_fetcher
from advanced_technical_analysis import technical_analyzer

# Test data fetching
data = data_fetcher.get_comprehensive_stock_data("AAPL", "1y")
print(f"Data keys: {data.keys()}")

# Test technical analysis
enhanced_data = technical_analyzer.calculate_all_indicators(data['history'])
print(f"Indicators calculated: {[col for col in enhanced_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]}")
```

### Performance Benchmarks
- Data loading: <2 seconds for 1-year data
- Technical analysis: <1 second for 20+ indicators
- Portfolio calculation: <3 seconds for 50+ holdings
- Chart rendering: <1 second for interactive charts

## Key Implementation Notes

### Enhanced Application Entry Points
1. **enhanced_app.py** - Primary enhanced application (recommended)
2. **Legacy applications** - app-py.py, app.py, globalstock.py (maintained for compatibility)

### Data Source Strategy
- **Primary**: yfinance for all data (price, fundamentals, news, options, etc.)
- **No external APIs**: Eliminates API key management and rate limiting issues
- **Comprehensive coverage**: Global markets, multiple asset classes

### Performance Optimizations
- **Streamlit caching**: Aggressive caching with appropriate TTL
- **Parallel processing**: Multi-threaded data fetching where applicable
- **Memory management**: Efficient data structures and garbage collection
- **Chart optimization**: Plotly charts optimized for interactivity and performance

### Error Recovery
- **Graceful degradation**: Application continues with partial data if some sources fail
- **User feedback**: Clear error messages and loading indicators
- **Fallback mechanisms**: Alternative data sources or reduced functionality when needed

## Development Commands

```bash
# Run enhanced application
streamlit run enhanced_app.py

# Install all dependencies
pip install -r requirements.txt

# Test individual modules
python -c "from enhanced_data_fetcher import data_fetcher; print('Data fetcher loaded successfully')"

# Development mode with auto-reload
streamlit run enhanced_app.py --server.runOnSave true
```

## Deployment Considerations

### Production Deployment
- **Streamlit Cloud**: Direct deployment from GitHub repository
- **Docker**: Containerized deployment for scalability
- **Memory requirements**: 2GB+ recommended for optimal performance
- **CPU**: Multi-core recommended for parallel processing

### Environment Variables
```bash
# Optional: Custom cache TTL
CACHE_TTL=3600

# Optional: Default risk-free rate for calculations
RISK_FREE_RATE=0.02
```