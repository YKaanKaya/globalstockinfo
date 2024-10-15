import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np

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
            'dividendYield': stock.info.get('dividendYield', 'N/A')
        }
        return info
    except Exception as e:
        st.error(f"Error fetching company info for {ticker}: {str(e)}")
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

    fig = go.Figure()
    colors = {'totalEsg': 'purple', 'environmentScore': 'green', 'socialScore': 'blue', 'governanceScore': 'orange'}

    for metric in numeric_data.index:
        fig.add_trace(go.Bar(
            x=[metric],
            y=[numeric_data.loc[metric].values[0]],
            name=metric,
            marker_color=colors.get(metric, 'gray')
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
        st.metric("Full Time Employees", f"{info.get('fullTimeEmployees', 'N/A'):,}")
    with col2:
        st.metric("Industry", info.get('industry', 'N/A'))
        st.metric("Country", info.get('country', 'N/A'))
    
    st.subheader("Financial Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}" if isinstance(info.get('marketCap'), (int, float)) else 'N/A')
    with col2:
        st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A')
    with col3:
        st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A'):.2%}" if isinstance(info.get('dividendYield'), (int, float)) else 'N/A')

def get_sentiment_score(news):
    try:
        sentiment_scores = []
        for article in news:
            blob = TextBlob(article['title'])
            sentiment_scores.append(blob.sentiment.polarity)
        return np.mean(sentiment_scores)
    except Exception as e:
        st.error(f"Error calculating sentiment for news: {str(e)}")
        return None

def display_sentiment_trend(news):
    sentiment_data = []
    for article in news:
        sentiment = TextBlob(article['title']).sentiment.polarity
        date = datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d')
        sentiment_data.append({'date': date, 'sentiment': sentiment})
    
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df = sentiment_df.groupby('date').mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sentiment_df.index, y=sentiment_df['sentiment'], mode='lines', name='Sentiment Score'))
    fig.update_layout(title="Sentiment Trend", yaxis_title="Average Sentiment Score", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

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
        stock = yf.Ticker(ticker)
        news = stock.news if hasattr(stock, 'news') else None
        sentiment_score = get_sentiment_score(news) if news else None

    if stock_data is not None and not stock_data.empty:
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}", 
                    f"{stock_data['Daily Return'].iloc[-1]:.2%}")
        col2.metric("50-Day MA", f"${stock_data['MA50'].iloc[-1]:.2f}")
        col3.metric("200-Day MA", f"${stock_data['MA200'].iloc[-1]:.2f}")
        if esg_data is not None:
            col4.metric("ESG Score", f"{esg_data.loc['totalEsg'].values[0]:.2f}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Chart", "🌿 ESG Analysis", "ℹ️ Company Info", "📰 Sentiment Analysis"])

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
            if news:
                st.subheader("Latest News Sentiment Analysis")
                display_sentiment_trend(news)
                if sentiment_score is not None:
                    st.metric("Sentiment Score", f"{sentiment_score:.2f}")
            else:
                st.warning("No recent news available for sentiment analysis.")

    else:
        st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

    st.markdown("---")
    st.markdown("Data provided by Yahoo Finance. This dashboard is for informational purposes only.")

if __name__ == "__main__":
    main()
