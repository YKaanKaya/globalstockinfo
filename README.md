# Enhanced Stock Analysis Dashboard

A comprehensive stock analysis dashboard built with Streamlit that provides in-depth analysis of stocks including technical indicators, financial statements, sentiment analysis, and portfolio tracking.

![Dashboard Preview](https://img.icons8.com/fluency/96/000000/stock-market.png)

## Features

- **Stock Data Visualization**: Interactive price charts with candlestick patterns and volume analysis
- **Technical Analysis**: Multiple technical indicators including RSI, MACD, Bollinger Bands
- **Fundamental Analysis**: Access to financial statements and key financial ratios
- **News Sentiment Analysis**: Analyze sentiment from recent news and its potential impact on stock price
- **Stock Recommendations**: Get automated stock recommendations based on multiple factors
- **Portfolio Tracking**: Manage your investment portfolio and track performance
- **Competitor Analysis**: Compare stock performance with industry peers
- **ESG Analysis**: Environmental, Social, and Governance scores visualization
- **Analyst Sentiment**: View analyst recommendations and consensus

## Project Structure

The project is organized into modular components for better maintenance:

```
enhanced-stock-dashboard/
├── app.py                       # Main application file
├── data_fetcher.py              # Data acquisition module
├── technical_analysis.py        # Technical analysis functions
├── visualizations.py            # Chart and visualization functions
├── financial_analysis.py        # Financial statement analysis
├── sentiment_analysis.py        # News and sentiment analysis
├── recommendation_engine.py     # Stock recommendation system
├── portfolio_tracker.py         # Portfolio management module
├── requirements.txt             # Required packages
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-stock-dashboard.git
cd enhanced-stock-dashboard
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your API keys:
```toml
A_KEY = "your_alpha_vantage_api_key"
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

### Analyzing a Stock

1. Enter a stock ticker in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. Select the time period for analysis
3. Choose technical indicators to display
4. Navigate through the tabs to view different aspects of the analysis:
   - Price Analysis: Stock price chart and returns analysis
   - Technical Indicators: RSI, MACD, and other technical indicators
   - ESG Analysis: Environmental, Social, and Governance scores
   - Company Info: Overview, financials, and analyst recommendations
   - News & Sentiment: Recent news with sentiment analysis
   - Financials: Income statement, balance sheet, and cash flow

### Portfolio Management

1. Go to the Portfolio Management tab
2. Add transactions by entering ticker, transaction type, shares, and price
3. View your portfolio summary, holdings, and performance metrics
4. Export your portfolio data for backup or import previously saved data
5. Maintain a watchlist of stocks you're interested in

## API Keys

This application uses the following APIs:

- **Alpha Vantage**: For financial statements and analyst estimates. Get a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Yahoo Finance**: Used for historical price data, company information, and ESG data (no API key required)

## Dependencies

The application requires the following main packages:

- streamlit
- pandas
- yfinance
- plotly
- requests
- textblob
- numpy

For a complete list, see `requirements.txt`.

## Performance Optimizations

- API calls are cached to reduce redundant requests
- Parallel data fetching where appropriate
- Retry logic for resilient API communication
- Intelligent data prefetching for common stocks

## Customization

You can customize the dashboard by modifying the following:

- Default ticker and time period in `app.py`
- Technical indicators list in `app.py`
- Styling in the `setup_page` function
- Recommendation factors and weights in `recommendation_engine.py`

## Limitations

- Free API tiers have request limits
- Some data might not be available for all stocks
- Financial statement data may be delayed

## Future Enhancements

- Machine learning-based price prediction
- Options analysis
- Excel/CSV export of analysis
- Mobile-optimized view
- Backtesting of trading strategies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
