import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from polygon import RESTClient
from datetime import datetime, timedelta

# Set up the Polygon client with your API key
polygon_client = RESTClient("VGoBXkRG21g0xrLvvSSYoGaabdhi36O6")

def get_stock_data(ticker, from_date, to_date):
    try:
        # Fetch daily close data
        resp = polygon_client.stocks_equities_aggregates(ticker, 1, "day", from_date, to_date, unadjusted=False)
        
        # Convert to DataFrame
        df = pd.DataFrame(resp.results)
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('date')
        df = df.rename(columns={'c': 'Close', 'h': 'High', 'l': 'Low', 'o': 'Open', 'v': 'Volume'})
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def compute_returns(data):
    data['Daily Return'] = data['Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

def get_esg_data_with_headers_and_error_handling(ticker):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://finance.yahoo.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    url = f"https://finance.yahoo.com/quote/{ticker}/sustainability?p={ticker}"
    
    # Add a random delay between requests
    time.sleep(random.uniform(1, 3))
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.RequestException as e:
        print(f"Failed to fetch data for {ticker}. Error: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    result = {}

    try:
        esg_section = soup.find('div', {'data-test': 'qsp-sustainability'})
        if not esg_section:
            print(f"ESG section not found for {ticker}")
            return None

        total_esg_risk = esg_section.find('div', string='Total ESG Risk')
        if total_esg_risk:
            score = total_esg_risk.find_next('div').text.strip()
            result["Total ESG risk score"] = float(score) if score.replace('.', '').isdigit() else None

        for category in ['Environment Risk', 'Social Risk', 'Governance Risk']:
            category_elem = esg_section.find('span', string=category)
            if category_elem:
                score = category_elem.find_next('span').text.strip()
                result[f"{category.split()[0].lower()} risk score"] = float(score) if score.replace('.', '').isdigit() else None

        controversy_level = esg_section.find('div', string='Controversy Level')
        if controversy_level:
            level = controversy_level.find_next('div').text.strip()
            result["Controversy level"] = int(level) if level.isdigit() else None

    except Exception as e:
        print(f"Error parsing ESG data for {ticker}: {e}")

    return result

def map_esg_risk_to_level(score):
    if score is None or not isinstance(score, (int, float)):
        return "Unknown"
    if score < 10:
        return "Very Low"
    elif 10 <= score < 20:
        return "Low"
    elif 20 <= score < 30:
        return "Medium"
    elif 30 <= score < 40:
        return "High"
    else:
        return "Severe"

def display_stock_chart(data, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Candlestick chart for stock prices
    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Price'), row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='200 Day MA', line=dict(color='red', width=1)), row=1, col=1)

    # Volume chart
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=False
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

def display_esg_chart(esg_data, ticker):
    categories = ['Environment', 'Social', 'Governance']
    scores = [esg_data.get(f'{cat.lower()} risk score', 0) for cat in categories]

    fig = go.Figure(data=[
        go.Bar(name='ESG Scores', x=categories, y=scores, text=scores, textposition='auto')
    ])

    fig.update_layout(
        title=f'{ticker} ESG Risk Scores',
        yaxis_title='Risk Score',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Financial Data Dashboard")
    st.markdown("This dashboard provides comprehensive stock analysis including price trends, returns, and ESG metrics.")

    # Sidebar for user input
    st.sidebar.header("Configure Your Analysis")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
    period = st.sidebar.selectbox("Select Time Period", 
                                  options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                                  format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
                                  index=3)

    # Convert period to date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(period[:-1]) * (30 if period[-1] == 'M' else 365))).strftime('%Y-%m-%d')

    # Fetch stock data
    with st.spinner('Fetching stock data...'):
        stock_data = get_stock_data(ticker, start_date, end_date)

    if stock_data is not None and not stock_data.empty:
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)

        # Display stock price chart
        st.header(f"{ticker} Stock Analysis")
        display_stock_chart(stock_data, ticker)

        # Display returns chart
        st.header(f"{ticker} Returns Analysis")
        display_returns_chart(stock_data, ticker)

        # Key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}", 
                    f"{stock_data['Daily Return'].iloc[-1]:.2%}")
        col2.metric("50-Day MA", f"${stock_data['MA50'].iloc[-1]:.2f}")
        col3.metric("200-Day MA", f"${stock_data['MA200'].iloc[-1]:.2f}")

        # Fetch and display ESG data
        with st.spinner('Fetching ESG data...'):
            esg_data = get_esg_data_with_headers_and_error_handling(ticker)

        if esg_data:
            st.header(f"{ticker} ESG Analysis")
            display_esg_chart(esg_data, ticker)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total ESG Risk", f"{esg_data['Total ESG risk score']:.2f}", 
                        map_esg_risk_to_level(esg_data['Total ESG risk score']))
            col2.metric("Environment Risk", f"{esg_data['environment risk score']:.2f}")
            col3.metric("Social Risk", f"{esg_data['social risk score']:.2f}")
            col4.metric("Governance Risk", f"{esg_data['governance risk score']:.2f}")
        else:
            st.warning("ESG data not available for this stock.")
    else:
        st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
