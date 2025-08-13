# 🚀 Enhanced Stock Analysis Dashboard

A comprehensive, professional-grade stock analysis dashboard built entirely with **yfinance** and **Streamlit**. This application provides advanced technical analysis, portfolio management, financial statement analysis, and interactive visualizations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Key Features

### 🔍 **Advanced Technical Analysis**
- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ADX, Ichimoku Cloud, Williams %R, CCI, ROC, Momentum, and more
- **Chart Pattern Detection**: Double tops/bottoms, head and shoulders, triangles, flags, pennants
- **Support/Resistance Levels**: Automated identification of key price levels
- **Trading Signals**: Real-time buy/sell/neutral signals based on multiple indicators
- **Custom Timeframes**: From 1 day to 10 years of historical data

### 💼 **Professional Portfolio Management**
- **Advanced Risk Metrics**: Sharpe ratio, Sortino ratio, Maximum Drawdown, VaR, CVaR
- **Performance Analytics**: Alpha, Beta, Information ratio, tracking error
- **Portfolio Optimization**: Sector allocation analysis and rebalancing suggestions
- **Transaction Tracking**: Complete buy/sell history with cost basis calculations
- **Benchmark Comparison**: Compare against S&P 500 and other indices
- **Tax Loss Harvesting**: Automated identification of tax optimization opportunities

### 💰 **Comprehensive Financial Analysis**
- **Financial Statement Analysis**: Income statement, balance sheet, cash flow analysis
- **Financial Health Scoring**: Proprietary 100-point scoring system
- **Growth Trend Analysis**: Revenue, earnings, and cash flow growth patterns
- **Financial Ratios**: 15+ key ratios including profitability, liquidity, and efficiency metrics
- **Stock Screening**: Multi-criteria screening with customizable filters
- **Company Comparison**: Side-by-side financial metrics comparison

### 📊 **Interactive Visualizations**
- **Modern Candlestick Charts**: With volume and technical indicators overlay
- **Risk-Return Analysis**: Interactive scatter plots with efficient frontier
- **Correlation Heatmaps**: Asset correlation analysis
- **Drawdown Charts**: Portfolio drawdown visualization
- **Options Analysis**: Options chain analysis with IV visualization
- **Sector Allocation**: Interactive pie charts and treemaps

### 🌍 **Market Coverage**
- **Global Stocks**: All stocks available on Yahoo Finance
- **Market Indices**: S&P 500, NASDAQ, Dow Jones, international indices
- **Sector ETFs**: Complete sector analysis and comparison
- **Cryptocurrencies**: Bitcoin, Ethereum, and major altcoins
- **Currency Pairs**: Major forex pairs
- **Commodities**: Gold, oil, and other commodities

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YKaanKaya/globalstockinfo.git
   cd globalstockinfo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the enhanced application:**
   ```bash
   streamlit run enhanced_app.py
   ```

4. **Access the dashboard:**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
enhanced-stock-dashboard/
├── enhanced_app.py                    # Main application file
├── enhanced_data_fetcher.py           # Comprehensive data acquisition
├── advanced_technical_analysis.py    # Technical indicators & patterns
├── advanced_portfolio_manager.py     # Portfolio analytics & management
├── enhanced_visualizations.py        # Interactive charts & visualizations
├── advanced_financial_analysis.py    # Financial statement analysis
├── requirements.txt                   # Python dependencies
├── ENHANCED_README.md                # This file
├── CLAUDE.md                         # Development guidance
└── legacy/                           # Original application files
    ├── app-py.py
    ├── globalstock.py
    └── ...
```

## 🎯 Usage Guide

### Stock Analysis
1. **Enter a ticker symbol** in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. **Select time period** (1D to 10Y)
3. **Choose technical indicators** to display
4. **Navigate tabs** for different analysis types

### Portfolio Management
1. **Add transactions** using the portfolio tab
2. **View performance metrics** and risk analytics
3. **Analyze sector allocation** and diversification
4. **Track gains/losses** with detailed position analysis

### Financial Analysis
1. **View financial health score** (0-100 scale)
2. **Analyze growth trends** and profitability metrics
3. **Compare financial ratios** across time periods
4. **Review cash flow** and balance sheet strength

### Stock Comparison
1. **Select multiple stocks** for comparison
2. **View normalized performance** charts
3. **Compare key metrics** side-by-side
4. **Analyze correlations** and relationships

## 🔧 Advanced Features

### Technical Analysis
- **Fibonacci Retracements**: Automatic calculation of key levels
- **Parabolic SAR**: Trend-following indicator with stop-loss levels
- **Volume Indicators**: OBV, VWAP, A/D Line, Chaikin Money Flow
- **Custom Indicators**: Easy to add new technical indicators

### Portfolio Analytics
- **Modern Portfolio Theory**: Efficient frontier and optimal allocation
- **Monte Carlo Simulation**: Risk modeling and scenario analysis
- **Backtesting**: Historical performance testing
- **Risk Budgeting**: Position sizing based on risk contribution

### Data Export
- **CSV Export**: Portfolio and analysis data
- **PDF Reports**: Automated report generation
- **Excel Integration**: Formatted spreadsheet exports
- **JSON API**: Data access for external applications

## 🎨 Customization

### Adding New Indicators
```python
# In advanced_technical_analysis.py
@staticmethod
def custom_indicator(data: pd.Series, window: int) -> pd.Series:
    """Your custom indicator logic here."""
    return data.rolling(window=window).mean()  # Example
```

### Modifying UI Theme
```python
# In enhanced_app.py - Custom CSS section
st.markdown("""
<style>
    .your-custom-class {
        /* Your styling here */
    }
</style>
""", unsafe_allow_html=True)
```

### Adding New Metrics
```python
# In advanced_portfolio_manager.py
def calculate_custom_metric(self, returns: pd.Series) -> float:
    """Your custom portfolio metric."""
    return your_calculation(returns)
```

## 📊 Performance Optimizations

- **Streamlit Caching**: Automatic caching of API calls and calculations
- **Parallel Processing**: Multi-threaded data fetching
- **Memory Optimization**: Efficient data structures and garbage collection
- **Load Balancing**: Smart API request distribution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- **Machine Learning Models**: Price prediction and anomaly detection
- **Real-time Data**: WebSocket integration for live quotes
- **Mobile App**: React Native mobile application
- **API Endpoints**: RESTful API for external integrations
- **Database Integration**: PostgreSQL/MongoDB for data persistence
- **User Authentication**: Multi-user support with accounts
- **Advanced Options**: Greeks calculation and options strategies
- **ESG Scoring**: Environmental, social, and governance analysis

## 🔒 Data Privacy

This application runs entirely on your local machine. No data is sent to external servers except for the necessary API calls to Yahoo Finance for market data.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, suggestions, or issues:
- Create an issue on GitHub
- Check the documentation in `CLAUDE.md`
- Review the code comments for implementation details

---

**Built with ❤️ using Python, Streamlit, and yfinance**

*Last updated: August 2025*