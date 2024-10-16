import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np

# Alpha Vantage API key from Streamlit secrets
api_key = st.secrets["A_KEY"]

def format_large_number(value):
    """Format large numbers into readable strings with suffixes."""
    if not isinstance(value, (int, float)):
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

@st.cache_data(ttl=3600)
def get_sp500_companies():
    """Fetch the list of S&P 500 companies from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        df.columns = ['Ticker', 'Company', 'Sector', 'Industry']
        return df
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies list: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data available for {ticker} in the selected date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    """Fetch company information using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extract relevant information
        company_info = {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'fullTimeEmployees': info.get('fullTimeEmployees', 'N/A'),
            'country': info.get('country', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'forwardPE': info.get('forwardPE', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A'),
            'longBusinessSummary': info.get('longBusinessSummary', 'N/A'),
            'website': info.get('website', 'N/A'),
            'address1': info.get('address1', ''),
            'city': info.get('city', ''),
            'state': info.get('state', ''),
            'zip': info.get('zip', ''),
            'phone': info.get('phone', 'N/A'),
        }
        return company_info
    except Exception as e:
        st.error(f"Error fetching company info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_esg_data(ticker):
    """Fetch ESG data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        if esg_data is None or esg_data.empty:
            st.warning(f"No ESG data found for {ticker}")
            return None
        return esg_data
    except Exception as e:
        st.error(f"Error fetching ESG data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_competitors(ticker):
    """Identify competitors within the same industry from S&P 500 list."""
    try:
        # Get the industry of the selected ticker
        company_info = get_company_info(ticker)
        if not company_info or company_info['industry'] == 'N/A':
            st.warning(f"Industry information not available for {ticker}")
            return []
        industry = company_info['industry']

        # Get the list of S&P 500 companies
        sp500_df = get_sp500_companies()
        if sp500_df is None or sp500_df.empty:
            st.warning("Could not retrieve S&P 500 companies.")
            return []

        # Filter companies in the same industry
        competitors_df = sp500_df[sp500_df['Industry'] == industry]
        # Exclude the selected ticker
        competitors_df = competitors_df[competitors_df['Ticker'] != ticker]

        # Get the list of competitor tickers
        competitors = competitors_df['Ticker'].tolist()[:5]  # Return top 5 competitors
        return competitors
    except Exception as e:
        st.error(f"Error fetching competitors for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def compare_performance(ticker, competitors):
    """Compare cumulative returns of the ticker with its competitors over 1 year."""
    try:
        if not competitors:
            st.warning("No competitors found for comparison.")
            return None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download([ticker] + competitors, start=start_date, end=end_date)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if data.empty:
            st.warning("No competitor data available.")
            return None
        returns = (data.pct_change() + 1).cumprod()
        return returns
    except Exception as e:
        st.error(f"Error comparing performance: {str(e)}")
        return None

def create_comparison_chart(comparison_data):
    """Create a Plotly chart comparing cumulative returns."""
    if comparison_data is None or comparison_data.empty:
        st.warning("No data available for comparison.")
        return None

    fig = go.Figure()
    for column in comparison_data.columns:
        fig.add_trace(go.Scatter(x=comparison_data.index, y=comparison_data[column], mode='lines', name=column))
    fig.update_layout(title="1 Year Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
    return fig

@st.cache_data(ttl=3600)
def get_news(ticker):
    """Fetch latest news articles for the ticker."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return None

def display_news(news):
    """Display the latest news articles."""
    st.subheader("Latest News")
    if not news:
        st.warning("No news available.")
        return
    for article in news[:5]:  # Display top 5 news articles
        st.markdown(f"### [{article['title']}]({article['link']})")
        try:
            pub_time = datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pub_time = "N/A"
        st.markdown(f"*Published on: {pub_time}*")
        st.markdown("---")

@st.cache_data(ttl=3600)
def get_sentiment_score(ticker):
    """Calculate sentiment score based on news headlines."""
    try:
        news = get_news(ticker)
        if not news:
            return "Neutral"
        sentiment_scores = []
        for article in news[:10]:
            blob = TextBlob(article['title'])
            sentiment_scores.append(blob.sentiment.polarity)
        average_score = np.mean(sentiment_scores)
        if average_score > 0.1:
            return "Positive"
        elif average_score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error calculating sentiment for {ticker}: {str(e)}")
        return "Neutral"

@st.cache_data(ttl=3600)
def get_analyst_estimates(ticker):
    """Fetch analyst estimates data from Alpha Vantage."""
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "ANALYST_ESTIMATES",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "analystEstimates" not in data:
            st.warning(f"No analyst estimates data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            if recommendations is None or recommendations.empty:
                st.warning(f"No analyst recommendations available for {ticker} from yfinance.")
                return None
            # Process recommendations
            consensus = {
                'Buy': recommendations['To Grade'].str.lower().isin(['buy', 'strong buy']).sum(),
                'Hold': recommendations['To Grade'].str.lower().isin(['hold']).sum(),
                'Sell': recommendations['To Grade'].str.lower().isin(['sell', 'strong sell']).sum()
            }
            return consensus
        else:
            estimates = data["analystEstimates"]
            if "recommendationKey" not in estimates.columns:
                st.warning(f"'recommendationKey' column not found in analyst estimates for {ticker} from Alpha Vantage.")
                return None
            recommendations = estimates['recommendationKey'].dropna().str.lower()
            consensus = {
                'Buy': recommendations.isin(['buy', 'strong buy']).sum(),
                'Hold': recommendations.isin(['hold']).sum(),
                'Sell': recommendations.isin(['sell', 'strong sell']).sum()
            }
            return consensus
    except Exception as e:
        st.error(f"Error fetching analyst estimates for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            if recommendations is None or recommendations.empty:
                st.warning(f"No analyst recommendations available for {ticker} from yfinance.")
                return None
            # Process recommendations
            consensus = {
                'Buy': recommendations['To Grade'].str.lower().isin(['buy', 'strong buy']).sum(),
                'Hold': recommendations['To Grade'].str.lower().isin(['hold']).sum(),
                'Sell': recommendations['To Grade'].str.lower().isin(['sell', 'strong sell']).sum()
            }
            return consensus
        except Exception as y:
            st.error(f"Error fetching analyst estimates for {ticker} from yfinance: {str(y)}")
            return None

def display_analyst_recommendations(consensus):
    """Display analyst recommendations as a pie chart."""
    if consensus is None:
        st.warning("No analyst consensus available.")
        return
    labels = list(consensus.keys())
    values = list(consensus.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Analyst Recommendations Consensus")
    st.plotly_chart(fig, use_container_width=True)

def compute_returns(data):
    """Compute daily and cumulative returns."""
    data['Daily Return'] = data['Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    """Compute moving averages."""
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

def display_stock_chart(data, ticker):
    """Display stock price chart with moving averages and volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Price'), row=1, col=1)

    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='orange', width=1)), row=1, col=1)
    if 'MA200' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='200 Day MA', line=dict(color='red', width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=2, col=1)

    fig.update_layout(
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def display_returns_chart(data, ticker):
    """Display cumulative returns chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Return'], mode='lines', name='Cumulative Return'))
    fig.update_layout(
        title=f'{ticker} Cumulative Returns',
        yaxis_title='Cumulative Return',
        xaxis_title='Date',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def display_esg_data(esg_data):
    """Display ESG scores as a bar chart."""
    st.subheader("ESG Data")
    relevant_metrics = ['totalEsg', 'environmentScore', 'socialScore', 'governanceScore']
    numeric_data = esg_data[esg_data.index.isin(relevant_metrics)]

    # Mapping for better metric names
    metric_names = {
        'totalEsg': 'Total ESG Score',
        'environmentScore': 'Environmental Score',
        'socialScore': 'Social Score',
        'governanceScore': 'Governance Score'
    }

    fig = go.Figure()
    colors = {'Total ESG Score': 'purple', 'Environmental Score': 'green', 'Social Score': 'blue', 'Governance Score': 'orange'}

    for metric in numeric_data.index:
        readable_metric = metric_names.get(metric, metric)
        fig.add_trace(go.Bar(
            x=[readable_metric],
            y=[numeric_data.loc[metric].values[0]],
            name=readable_metric,
            marker_color=colors.get(readable_metric, 'gray')
        ))

    fig.update_layout(
        title="ESG Scores",
        yaxis_title="Score",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def display_company_info(info):
    """Display company information with metrics."""
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sector", info.get('sector', 'N/A'))
        if isinstance(info.get('fullTimeEmployees'), (int, float)):
            st.metric("Full Time Employees", f"{int(info.get('fullTimeEmployees')):,}")
        else:
            st.metric("Full Time Employees", "N/A")
    with col2:
        st.metric("Industry", info.get('industry', 'N/A'))
        st.metric("Country", info.get('country', 'N/A'))

    st.subheader("Financial Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", format_large_number(info.get('marketCap')))
    with col2:
        forward_pe = info.get('forwardPE', 'N/A')
        forward_pe_display = f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else 'N/A'
        st.metric("Forward P/E", forward_pe_display)
    with col3:
        dividend_yield = info.get('dividendYield', 'N/A')
        dividend_yield_display = f"{dividend_yield:.2%}" if isinstance(dividend_yield, (int, float)) else 'N/A'
        st.metric("Dividend Yield", dividend_yield_display)

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', 'N/A'))

    st.subheader("Contact Information")
    if info.get('website', 'N/A') != 'N/A':
        st.markdown(f"**Website:** [{info.get('website')}]({info.get('website')})")
    else:
        st.markdown(f"**Website:** N/A")
    st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    address_parts = [info.get('address1', ''), info.get('city', ''), info.get('state', ''), info.get('zip', ''), info.get('country', '')]
    address = ', '.join(part for part in address_parts if part)
    st.markdown(f"**Address:** {address}")

def display_rsi_chart(data):
    """Display RSI chart."""
    if 'RSI' not in data.columns:
        st.warning("RSI data not available.")
        return
    st.subheader("Relative Strength Index (RSI)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name='RSI'
    ))
    fig.update_layout(
        title="RSI Over Time",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_recommendation(ticker, company_info, esg_data, sentiment_score, data, analyst_consensus):
    """Generate stock recommendation based on various factors."""
    score = 0
    factors = {}

    # P/E Ratio
    forward_pe = company_info.get('forwardPE', None)
    if isinstance(forward_pe, (int, float)) and forward_pe != 0:
        if forward_pe < 15:
            factors['P/E Ratio'] = 'Positive'
            score += 1
        elif 15 <= forward_pe <= 25:
            factors['P/E Ratio'] = 'Neutral'
        else:
            factors['P/E Ratio'] = 'Negative'
            score -= 1
    else:
        factors['P/E Ratio'] = 'Neutral'

    # Dividend Yield
    dividend_yield = company_info.get('dividendYield', None)
    if isinstance(dividend_yield, (int, float)):
        if dividend_yield > 0.03:
            factors['Dividend Yield'] = 'Positive'
            score += 1
        elif 0.01 <= dividend_yield <= 0.03:
            factors['Dividend Yield'] = 'Neutral'
        else:
            factors['Dividend Yield'] = 'Negative'
            score -= 1
    else:
        factors['Dividend Yield'] = 'Neutral'

    # ESG Score
    if esg_data is not None:
        esg_score = esg_data.loc['totalEsg'].values[0]
        if esg_score > 50:
            factors['ESG Score'] = 'Positive'
            score += 1
        elif 30 <= esg_score <= 50:
            factors['ESG Score'] = 'Neutral'
        else:
            factors['ESG Score'] = 'Negative'
            score -= 1
    else:
        factors['ESG Score'] = 'Neutral'

    # Sentiment Score
    if sentiment_score == 'Positive':
        factors['Sentiment'] = 'Positive'
        score += 1
    elif sentiment_score == 'Neutral':
        factors['Sentiment'] = 'Neutral'
    elif sentiment_score == 'Negative':
        factors['Sentiment'] = 'Negative'
        score -= 1
    else:
        factors['Sentiment'] = 'Neutral'

    # RSI Indicator
    if 'RSI' in data.columns:
        latest_rsi = data['RSI'].iloc[-1]
        if latest_rsi < 30:
            factors['RSI'] = 'Positive (Oversold)'
            score += 1
        elif 30 <= latest_rsi <= 70:
            factors['RSI'] = 'Neutral'
        else:
            factors['RSI'] = 'Negative (Overbought)'
            score -= 1
    else:
        factors['RSI'] = 'Neutral'

    # Analyst Consensus
    if analyst_consensus is not None:
        buy = analyst_consensus.get('Buy', 0)
        hold = analyst_consensus.get('Hold', 0)
        sell = analyst_consensus.get('Sell', 0)
        if buy > sell and buy > hold:
            factors['Analyst Consensus'] = 'Positive'
            score += 1
        elif sell > buy and sell > hold:
            factors['Analyst Consensus'] = 'Negative'
            score -= 1
        else:
            factors['Analyst Consensus'] = 'Neutral'
    else:
        factors['Analyst Consensus'] = 'Neutral'

    # Generate Recommendation
    if score >= 4:
        recommendation = 'Buy'
    elif score <= -1:
        recommendation = 'Sell'
    else:
        recommendation = 'Hold'

    return recommendation, factors

def display_recommendation_visualization(recommendation, factors):
    """Visualize recommendation factors and overall score."""
    # Convert factors to numerical scores
    factor_scores = []
    factor_names = []
    for factor, assessment in factors.items():
        if 'Positive' in assessment:
            score = 1
        elif 'Negative' in assessment:
            score = -1
        else:
            score = 0
        factor_names.append(factor)
        factor_scores.append(score)

    # Create a bar chart of factor scores
    fig = go.Figure(data=[
        go.Bar(x=factor_names, y=factor_scores, marker_color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in factor_scores])
    ])
    fig.update_layout(
        title="Factor Scores",
        xaxis_title="Factors",
        yaxis_title="Score",
        yaxis=dict(tickmode='linear', tick0=-1, dtick=1),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600)
def get_income_statement(ticker):
    """Fetch income statement data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch income statement data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"No income statement data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            financials = stock.financials
            if financials.empty:
                st.error(f"No income statement data available for {ticker} from yfinance.")
                return None
            income_statement = financials.transpose().reset_index()
            income_statement.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
            return income_statement
        else:
            income_statement = pd.DataFrame(data["annualReports"])
            income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
            return income_statement
    except Exception as e:
        st.error(f"Error fetching income statement data for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            if financials.empty:
                st.error(f"No income statement data available for {ticker} from yfinance.")
                return None
            income_statement = financials.transpose().reset_index()
            income_statement.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
            return income_statement
        except Exception as y:
            st.error(f"Error fetching income statement data for {ticker} from yfinance: {str(y)}")
            return None

def display_income_statement(income_statement):
    """Display income statement data."""
    st.subheader("Income Statement")
    # Check if income_statement is valid
    if income_statement is None or income_statement.empty:
        st.warning("Income statement data not available.")
        return

    # Display the last 5 annual reports
    reports = income_statement.head(5)
    reports = reports.set_index('fiscalDateEnding')

    # Select relevant columns
    columns_to_display = ['totalRevenue', 'grossProfit', 'ebit', 'netIncome']

    # Check if columns exist
    missing_columns = [col for col in columns_to_display if col not in reports.columns]
    if missing_columns:
        st.warning(f"The following columns are missing in the income statement data: {', '.join(missing_columns)}")
        return

    reports = reports[columns_to_display]

    # Convert columns to numeric, coercing errors
    reports = reports.apply(pd.to_numeric, errors='coerce')

    # Transpose the DataFrame
    reports = reports.transpose()

    # Rename the index for better readability
    reports.index = ['Total Revenue', 'Gross Profit', 'EBIT', 'Net Income']

    # Apply formatting
    formatted_reports = reports.style.format("{:,.0f}")

    st.dataframe(formatted_reports)

@st.cache_data(ttl=3600)
def get_balance_sheet(ticker):
    """Fetch balance sheet data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch balance sheet data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"No balance sheet data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                st.error(f"No balance sheet data available for {ticker} from yfinance.")
                return None
            balance_sheet = balance_sheet.transpose().reset_index()
            balance_sheet.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
        else:
            balance_sheet = pd.DataFrame(data["annualReports"])
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
    except Exception as e:
        st.error(f"Error fetching balance sheet data for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                st.error(f"No balance sheet data available for {ticker} from yfinance.")
                return None
            balance_sheet = balance_sheet.transpose().reset_index()
            balance_sheet.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
            return balance_sheet
        except Exception as y:
            st.error(f"Error fetching balance sheet data for {ticker} from yfinance: {str(y)}")
            return None

def display_balance_sheet(balance_sheet):
    """Display balance sheet data."""
    st.subheader("Balance Sheet")
    # Check if balance_sheet is valid
    if balance_sheet is None or balance_sheet.empty:
        st.warning("Balance sheet data not available.")
        return

    reports = balance_sheet.head(5)
    reports = reports.set_index('fiscalDateEnding')

    # Select relevant columns
    columns_to_display = ['totalAssets', 'totalLiabilities', 'totalShareholderEquity']

    # Check if columns exist
    missing_columns = [col for col in columns_to_display if col not in reports.columns]
    if missing_columns:
        st.warning(f"The following columns are missing in the balance sheet data: {', '.join(missing_columns)}")
        return

    reports = reports[columns_to_display]

    # Convert columns to numeric, coercing errors
    reports = reports.apply(pd.to_numeric, errors='coerce')

    # Transpose the DataFrame
    reports = reports.transpose()

    # Rename the index for better readability
    reports.index = ['Total Assets', 'Total Liabilities', 'Total Shareholder Equity']

    # Apply formatting
    formatted_reports = reports.style.format("{:,.0f}")

    st.dataframe(formatted_reports)

@st.cache_data(ttl=3600)
def get_cash_flow(ticker):
    """Fetch cash flow data from Alpha Vantage or yfinance as fallback."""
    try:
        # Fetch cash flow data from Alpha Vantage
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "CASH_FLOW",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"No cash flow data available for {ticker} from Alpha Vantage. Attempting to fetch from yfinance.")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow
            if cash_flow.empty:
                st.error(f"No cash flow data available for {ticker} from yfinance.")
                return None
            cash_flow = cash_flow.transpose().reset_index()
            cash_flow.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
        else:
            cash_flow = pd.DataFrame(data["annualReports"])
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
    except Exception as e:
        st.error(f"Error fetching cash flow data for {ticker}: {str(e)}")
        # Attempt to fetch from yfinance as a secondary fallback
        try:
            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow
            if cash_flow.empty:
                st.error(f"No cash flow data available for {ticker} from yfinance.")
                return None
            cash_flow = cash_flow.transpose().reset_index()
            cash_flow.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            return cash_flow
        except Exception as y:
            st.error(f"Error fetching cash flow data for {ticker} from yfinance: {str(y)}")
            return None

def display_cash_flow(cash_flow):
    """Display cash flow statement data."""
    st.subheader("Cash Flow Statement")
    # Check if cash_flow is valid
    if cash_flow is None or cash_flow.empty:
        st.warning("Cash flow data not available.")
        return

    reports = cash_flow.head(5)
    reports = reports.set_index('fiscalDateEnding')

    # Select relevant columns
    columns_to_display = ['operatingCashflow', 'cashflowFromInvestment', 'cashflowFromFinancing', 'netIncome']

    # Check if columns exist
    missing_columns = [col for col in columns_to_display if col not in reports.columns]
    if missing_columns:
        st.warning(f"The following columns are missing in the cash flow data: {', '.join(missing_columns)}")
        return

    reports = reports[columns_to_display]

    # Convert columns to numeric, coercing errors
    reports = reports.apply(pd.to_numeric, errors='coerce')

    # Transpose the DataFrame
    reports = reports.transpose()

    # Rename the index for better readability
    reports.index = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Net Income']

    # Apply formatting
    formatted_reports = reports.style.format("{:,.0f}")

    st.dataframe(formatted_reports)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Enhanced Stock Analysis Dashboard")

    st.sidebar.title("Stock Analysis Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
    period = st.sidebar.selectbox("Select Time Period",
                                  options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                                  format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
                                  index=3)

    end_date = datetime.now().strftime('%Y-%m-%d')
    if period.endswith('M'):
        delta_days = int(period[:-1]) * 30
    else:
        delta_days = int(period[:-1]) * 365
    start_date = (datetime.now() - timedelta(days=delta_days)).strftime('%Y-%m-%d')

    st.title(f"{ticker} Enhanced Stock Analysis Dashboard")
    st.write(f"Analyzing data from {start_date} to {end_date}")

    with st.spinner('Fetching data...'):
        stock_data = get_stock_data(ticker, start_date, end_date)
        company_info = get_company_info(ticker)
        esg_data = get_esg_data(ticker)
        sentiment_score = get_sentiment_score(ticker)
        competitors = get_competitors(ticker)
        comparison_data = compare_performance(ticker, competitors)
        income_statement = get_income_statement(ticker)
        balance_sheet = get_balance_sheet(ticker)
        cash_flow = get_cash_flow(ticker)
        analyst_consensus = get_analyst_estimates(ticker)

    if stock_data is not None and not stock_data.empty:
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)
        stock_data = get_rsi(stock_data)

        # Generate Recommendation
        recommendation, factors = generate_recommendation(ticker, company_info, esg_data, sentiment_score, stock_data, analyst_consensus)

        # Display Top Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}",
                    f"{stock_data['Daily Return'].iloc[-1]:.2%}")
        col2.metric("50-Day MA", f"${stock_data['MA50'].iloc[-1]:.2f}")
        col3.metric("200-Day MA", f"${stock_data['MA200'].iloc[-1]:.2f}")
        if esg_data is not None:
            esg_score = esg_data.loc['totalEsg'].values[0]
            col4.metric("ESG Score", f"{esg_score:.2f}")
        else:
            col4.metric("ESG Score", "N/A")
        if sentiment_score is not None:
            col5.metric("Sentiment", sentiment_score)
        else:
            col5.metric("Sentiment", "N/A")
        col6.metric("Recommendation", recommendation)

        # Create Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Stock Chart", "🌿 ESG Analysis", "ℹ️ Company Info", "📰 News & Sentiment", "🔍 Unique Insights", "📊 Financials"])

        # Tab 1: Stock Chart
        with tab1:
            st.header("Stock Price Analysis")
            display_stock_chart(stock_data, ticker)
            st.header("Returns Analysis")
            display_returns_chart(stock_data, ticker)
            st.header("Technical Indicators")
            display_rsi_chart(stock_data)

        # Tab 2: ESG Analysis
        with tab2:
            if esg_data is not None:
                display_esg_data(esg_data)
            else:
                st.warning("ESG data not available for this stock.")

        # Tab 3: Company Info
        with tab3:
            if company_info:
                display_company_info(company_info)
            else:
                st.warning("Company information not available.")

        # Tab 4: News & Sentiment
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                news = get_news(ticker)
                if news:
                    display_news(news)
                else:
                    st.warning("No recent news available for this stock.")
            with col2:
                st.subheader("Sentiment Analysis")
                st.write(f"The overall sentiment based on recent news headlines is **{sentiment_score}**.")

        # Tab 5: Unique Insights
        with tab5:
            st.header("Unique Insights")

            st.subheader("Automated Stock Recommendation")
            st.write("**Recommendation:**", recommendation)
            st.write("**Disclaimer:** This recommendation is generated automatically based on predefined criteria and is not financial advice. This app is intended for improving technical skills and sharing them with potential interested parties.")

            st.subheader("Factors Considered")
            for factor, assessment in factors.items():
                st.write(f"- **{factor}**: {assessment}")

            # Visualization for Recommendation
            st.subheader("Recommendation Visualization")
            display_recommendation_visualization(recommendation, factors)

            # Competitor Comparison
            if comparison_data is not None and not comparison_data.empty:
                st.subheader("Competitor Comparison")
                st.plotly_chart(create_comparison_chart(comparison_data), use_container_width=True)
            else:
                st.warning("Competitor comparison data not available.")

            # Analyst Recommendations
            st.subheader("Analyst Recommendations")
            display_analyst_recommendations(analyst_consensus)

        # Tab 6: Financials
        with tab6:
            st.header("Financial Statements")
            fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
            with fin_tab1:
                if income_statement is not None:
                    display_income_statement(income_statement)
                else:
                    st.warning("Income statement data not available.")
            with fin_tab2:
                if balance_sheet is not None:
                    display_balance_sheet(balance_sheet)
                else:
                    st.warning("Balance sheet data not available.")
            with fin_tab3:
                if cash_flow is not None:
                    display_cash_flow(cash_flow)
                else:
                    st.warning("Cash flow data not available.")

    if __name__ == "__main__":
        main()

    st.markdown("---")
    st.markdown("Data provided by Yahoo Finance and Alpha Vantage. This dashboard is for informational purposes only.")
