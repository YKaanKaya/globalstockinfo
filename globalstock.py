import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import http.client
import json
import os

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

def get_rapidapi_esg_data(ticker):
    api_key = os.environ.get('RAPIDAPI_KEY')
    if not api_key:
        st.error("RapidAPI key not found. Please set the RAPIDAPI_KEY environment variable.")
        return None

    conn = http.client.HTTPSConnection("esg-risk-ratings-for-stocks.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "esg-risk-ratings-for-stocks.p.rapidapi.com"
    }
    try:
        conn.request("GET", f"/api/v1/resources/esg?ticker={ticker}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        esg_data = json.loads(data.decode("utf-8"))
        
        if esg_data and 'esg' in esg_data:
            return esg_data['esg']
        else:
            st.warning(f"No ESG data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"Error fetching RapidAPI ESG data for {ticker}: {str(e)}")
        return None

def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.warning(f"Error fetching company info for {ticker}: {str(e)}")
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

def display_esg_data(esg_data):
    st.subheader("ESG Risk Ratings")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total ESG Risk Score", f"{esg_data['totalEsgRiskScore']:.2f}")
    col2.metric("Environment Risk Score", f"{esg_data['environmentRiskScore']:.2f}")
    col3.metric("Social Risk Score", f"{esg_data['socialRiskScore']:.2f}")
    col4.metric("Governance Risk Score", f"{esg_data['governanceRiskScore']:.2f}")

    st.subheader("ESG Risk Category")
    st.write(esg_data['esgRiskCategory'])

    st.subheader("ESG Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Environment Performance", esg_data['environmentPerformance'])
    col2.metric("Social Performance", esg_data['socialPerformance'])
    col3.metric("Governance Performance", esg_data['governancePerformance'])

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Financial Data Dashboard")
    st.markdown("This dashboard provides comprehensive stock analysis including price trends, returns, ESG metrics, and company information.")

    st.sidebar.header("Configure Your Analysis")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
    period = st.sidebar.selectbox("Select Time Period", 
                                  options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                                  format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
                                  index=3)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(period[:-1]) * (30 if period[-1] == 'M' else 365))).strftime('%Y-%m-%d')

    st.write(f"Debug: Fetching data for {ticker} from {start_date} to {end_date}")

    with st.spinner('Fetching stock data...'):
        stock_data = get_stock_data(ticker, start_date, end_date)

    if stock_data is not None and not stock_data.empty:
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)

        st.header(f"{ticker} Stock Analysis")
        display_stock_chart(stock_data, ticker)

        st.header(f"{ticker} Returns Analysis")
        display_returns_chart(stock_data, ticker)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}", 
                    f"{stock_data['Daily Return'].iloc[-1]:.2%}")
        col2.metric("50-Day MA", f"${stock_data['MA50'].iloc[-1]:.2f}")
        col3.metric("200-Day MA", f"${stock_data['MA200'].iloc[-1]:.2f}")

        with st.spinner('Fetching ESG data...'):
            esg_data = get_rapidapi_esg_data(ticker)

        if esg_data:
            st.header(f"{ticker} ESG Analysis")
            display_esg_data(esg_data)
        else:
            st.warning("ESG data not available for this stock.")

        company_info = get_company_info(ticker)
        if company_info:
            st.subheader("Company Information")
            col1, col2 = st.columns(2)
            col1.metric("Sector", company_info.get('sector', 'N/A'))
            col2.metric("Industry", company_info.get('industry', 'N/A'))
            col1.metric("Full Time Employees", company_info.get('fullTimeEmployees', 'N/A'))
            col2.metric("Country", company_info.get('country', 'N/A'))
            
            st.subheader("Financial Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Market Cap", f"${company_info.get('marketCap', 'N/A'):,}")
            col2.metric("Forward P/E", company_info.get('forwardPE', 'N/A'))
            col3.metric("Dividend Yield", f"{company_info.get('dividendYield', 'N/A'):.2%}" if company_info.get('dividendYield') else 'N/A')
    else:
        st.error(f"Unable to fetch data for {ticker}. Please check the debug information above and try again.")

if __name__ == "__main__":
    main()
