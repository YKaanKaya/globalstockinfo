import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np

def format_large_number(value):
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
    # Fetch the list of S&P 500 companies from Wikipedia
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
def get_esg_data(ticker):
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
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = {
            'sector': stock.info.get('sector', 'N/A'),
            'industry': stock.info.get('industry', 'N/A'),
            'fullTimeEmployees': stock.info.get('fullTimeEmployees', 'N/A'),
            'country': stock.info.get('country', 'N/A'),
            'marketCap': stock.info.get('marketCap', 'N/A'),
            'forwardPE': stock.info.get('forwardPE', 'N/A'),
            'dividendYield': stock.info.get('dividendYield', 'N/A'),
            'longBusinessSummary': stock.info.get('longBusinessSummary', 'N/A'),
            'website': stock.info.get('website', 'N/A'),
            'address1': stock.info.get('address1', ''),
            'city': stock.info.get('city', ''),
            'state': stock.info.get('state', ''),
            'zip': stock.info.get('zip', ''),
            'phone': stock.info.get('phone', 'N/A'),
        }
        return info
    except Exception as e:
        st.error(f"Error fetching company info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_competitors(ticker):
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
    if comparison_data is None or comparison_data.empty:
        st.warning("No data available for comparison.")
        return None

    fig = go.Figure()
    for column in comparison_data.columns:
        fig.add_trace(go.Scatter(x=comparison_data.index, y=comparison_data[column], mode='lines', name=column))
    fig.update_layout(title="1 Year Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
    return fig

@st.cache_data(ttl=3600)
def get_innovation_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        r_and_d = stock.info.get('researchAndDevelopment', 0)
        revenue = stock.info.get('totalRevenue', 1)  # Avoid division by zero
        r_and_d_intensity = (r_and_d / revenue) * 100 if revenue else 0
        return {
            'R&D Spending': r_and_d,
            'R&D Intensity': r_and_d_intensity
        }
    except Exception as e:
        st.error(f"Error fetching innovation metrics for {ticker}: {str(e)}")
        return None

def create_innovation_chart(innovation_data):
    fig = go.Figure(data=[go.Bar(x=list(innovation_data.keys()), y=list(innovation_data.values()))])
    fig.update_layout(title="Innovation Metrics", xaxis_title="Metric", yaxis_title="Value")
    return fig

@st.cache_data(ttl=3600)
def get_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return None

def display_news(news):
    st.subheader("Latest News")
    for article in news[:5]:  # Display top 5 news articles
        st.write(f"**{article['title']}**")
        st.write(f"*{datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}*")
        st.write(article['link'])
        st.write("---")

@st.cache_data(ttl=3600)
def get_recommendations(ticker):
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            # Reset index to ensure 'Date' is a column
            recommendations.reset_index(inplace=True)
            # Determine which columns are present
            grade_columns = ['To Grade', 'Action', 'Firm']
            # Use 'To Grade' or 'Action' column for recommendations
            if 'To Grade' in recommendations.columns:
                latest_recommendations = recommendations[['To Grade']].tail(100)
                grade_column = 'To Grade'
            elif 'Action' in recommendations.columns:
                latest_recommendations = recommendations[['Action']].tail(100)
                grade_column = 'Action'
            else:
                st.warning("No suitable grade column found in recommendations.")
                return None

            # Map grades to categories
            mapping = {
                'Strong Buy': 'Strong Buy',
                'Buy': 'Buy',
                'Hold': 'Hold',
                'Sell': 'Sell',
                'Strong Sell': 'Strong Sell',
                'Underperform': 'Sell',
                'Outperform': 'Buy',
                'Neutral': 'Hold',
                'Market Perform': 'Hold',
                'Perform': 'Hold',
                'Overweight': 'Buy',
                'Underweight': 'Sell',
                'Equal-Weight': 'Hold',
                'Equal-weight': 'Hold',
                'Sector Perform': 'Hold',
                'Sector Outperform': 'Buy',
                'Positive': 'Buy',
                'Negative': 'Sell',
                'Mixed': 'Hold',
                # Add other mappings as necessary
            }
            latest_recommendations['Recommendation'] = latest_recommendations[grade_column].map(mapping)
            recommendation_counts = latest_recommendations['Recommendation'].value_counts()
            return recommendation_counts
        else:
            st.warning("No recommendations data available.")
            return None
    except Exception as e:
        st.error(f"Error fetching recommendations for {ticker}: {str(e)}")
        return None

def display_recommendations(recommendation_counts):
    if recommendation_counts is not None and not recommendation_counts.empty:
        st.subheader("Analyst Recommendations")
        
        categories = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        colors = ['red', 'lightcoral', 'gray', 'lightgreen', 'green']
        
        fig = go.Figure()
        for category, color in zip(categories, colors):
            count = recommendation_counts.get(category, 0)
            fig.add_trace(go.Bar(
                x=[category],
                y=[count],
                name=category,
                marker_color=color
            ))
        
        fig.update_layout(
            title="Analyst Recommendations (Last 100 Recommendations)",
            xaxis_title="Recommendation",
            yaxis_title="Number of Analysts",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Raw Recommendation Data:")
        st.dataframe(recommendation_counts)
    else:
        st.warning("No analyst recommendations available.")

@st.cache_data(ttl=3600)
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:10]  # Get latest 10 news items
        sentiment_scores = []
        for article in news:
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
        return None

def compute_returns(data):
    data['Daily Return'] = data['Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

def display_stock_chart(data, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='200 Day MA', line=dict(color='red', width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=2, col=1)

    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def display_returns_chart(data, ticker):
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
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sector", info.get('sector', 'N/A'))
        st.metric("Full Time Employees", f"{info.get('fullTimeEmployees', 'N/A'):,}" if isinstance(info.get('fullTimeEmployees'), (int, float)) else 'N/A')
    with col2:
        st.metric("Industry", info.get('industry', 'N/A'))
        st.metric("Country", info.get('country', 'N/A'))

    st.subheader("Financial Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", format_large_number(info.get('marketCap')))
    with col2:
        st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A')
    with col3:
        st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A'):.2%}" if isinstance(info.get('dividendYield'), (int, float)) else 'N/A')

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', 'N/A'))

    st.subheader("Contact Information")
    st.write(f"Website: {info.get('website', 'N/A')}")
    st.write(f"Phone: {info.get('phone', 'N/A')}")
    address_parts = [info.get('address1', ''), info.get('city', ''), info.get('state', ''), info.get('zip', ''), info.get('country', '')]
    address = ', '.join(part for part in address_parts if part)
    st.write(f"Address: {address}")

def main():
    st.set_page_config(layout="wide", page_title="Enhanced Stock Analysis Dashboard")

    st.sidebar.title("Stock Analysis Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
    period = st.sidebar.selectbox("Select Time Period",
                                  options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                                  format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
                                  index=3)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(period[:-1]) * (30 if period[-1] == 'M' else 365))).strftime('%Y-%m-%d')

    st.title(f"{ticker} Enhanced Stock Analysis Dashboard")
    st.write(f"Analyzing data from {start_date} to {end_date}")

    with st.spinner('Fetching data...'):
        stock_data = get_stock_data(ticker, start_date, end_date)
        esg_data = get_esg_data(ticker)
        company_info = get_company_info(ticker)
        news = get_news(ticker)
        recommendations = get_recommendations(ticker)
        sentiment_score = get_sentiment_score(ticker)
        competitors = get_competitors(ticker)
        comparison_data = compare_performance(ticker, competitors)
        innovation_data = get_innovation_metrics(ticker)

    if stock_data is not None and not stock_data.empty:
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)

        col1, col2, col3, col4, col5 = st.columns(5)
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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Stock Chart", "🌿 ESG Analysis", "ℹ️ Company Info", "📰 News & Recommendations", "🔍 Unique Insights"])

        with tab1:
            st.header("Stock Price Analysis")
            display_stock_chart(stock_data, ticker)
            st.header("Returns Analysis")
            display_returns_chart(stock_data, ticker)

        with tab2:
            if esg_data is not None:
                display_esg_data(esg_data)
            else:
                st.warning("ESG data not available for this stock.")

        with tab3:
            if company_info:
                display_company_info(company_info)
            else:
                st.warning("Company information not available.")

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                if news:
                    display_news(news)
                else:
                    st.warning("No recent news available for this stock.")

            with col2:
                if recommendations is not None and not recommendations.empty:
                    display_recommendations(recommendations)
                else:
                    st.warning("No analyst recommendations available for this stock.")

        with tab5:
            st.header("Unique Insights")

            if comparison_data is not None and not comparison_data.empty:
                st.subheader("Competitor Comparison")
                st.plotly_chart(create_comparison_chart(comparison_data), use_container_width=True)
            else:
                st.warning("Competitor comparison data not available.")

            if innovation_data:
                st.subheader("Innovation Metrics")
                col1, col2 = st.columns(2)
                col1.metric("R&D Spending", format_large_number(innovation_data['R&D Spending']))
                col2.metric("R&D Intensity", f"{innovation_data['R&D Intensity']:.2f}%")
                st.plotly_chart(create_innovation_chart(innovation_data), use_container_width=True)
            else:
                st.warning("Innovation metrics not available for this stock.")

    else:
        st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

    st.markdown("---")
    st.markdown("Data provided by Yahoo Finance and Wikipedia. This dashboard is for informational purposes only.")

if __name__ == "__main__":
    main()
