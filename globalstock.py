import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total ESG", f"{esg_data.loc['totalEsg'].values[0]:.2f}")
    col2.metric("Environment", f"{esg_data.loc['environmentScore'].values[0]:.2f}")
    col3.metric("Social", f"{esg_data.loc['socialScore'].values[0]:.2f}")
    col4.metric("Governance", f"{esg_data.loc['governanceScore'].values[0]:.2f}")

    st.subheader("Additional ESG Information")
    additional_info = {
        'ESG Performance': esg_data.loc['esgPerformance'].values[0],
        'Highest Controversy': esg_data.loc['highestControversy'].values[0],
        'Rating Year': esg_data.loc['ratingYear'].values[0],
        'Rating Month': esg_data.loc['ratingMonth'].values[0]
    }
    st.table(pd.DataFrame.from_dict(additional_info, orient='index', columns=['Value']))

    st.subheader("Raw ESG Data")
    st.dataframe(esg_data)

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

def get_recommendations(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.recommendations
    except Exception as e:
        st.error(f"Error fetching recommendations for {ticker}: {str(e)}")
        return None

def display_recommendations(recommendations):
    if recommendations is not None and not recommendations.empty:
        st.subheader("Analyst Recommendations")
        st.dataframe(recommendations.tail(10))  # Display last 10 recommendations
    else:
        st.warning("No analyst recommendations available.")

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Financial Data Dashboard")
    st.markdown("This dashboard provides comprehensive stock analysis including price trends, returns, ESG metrics, company information, news, and analyst recommendations.")

    st.sidebar.header("Configure Your Analysis")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
    period = st.sidebar.selectbox("Select Time Period", 
                                  options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                                  format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
                                  index=3)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(period[:-1]) * (30 if period[-1] == 'M' else 365))).strftime('%Y-%m-%d')

    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}")

    tab1, tab2, tab3, tab4 = st.tabs(["Stock Analysis", "ESG Analysis", "Company Info", "News & Recommendations"])

    with tab1:
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
        else:
            st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

    with tab2:
        with st.spinner('Fetching ESG data...'):
            esg_data = get_esg_data(ticker)

        if esg_data is not None:
            st.header(f"{ticker} ESG Analysis")
            display_esg_data(esg_data)
        else:
            st.warning("ESG data not available for this stock.")

    with tab3:
        with st.spinner('Fetching company information...'):
            company_info = get_company_info(ticker)

        if company_info:
            display_company_info(company_info)
        else:
            st.warning("Company information not available.")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner('Fetching news...'):
                news = get_news(ticker)
            if news:
                display_news(news)
            else:
                st.warning("No news available for this stock.")
        
        with col2:
            with st.spinner('Fetching recommendations...'):
                recommendations = get_recommendations(ticker)
            if recommendations is not None:
                display_recommendations(recommendations)
            else:
                st.warning("No recommendations available for this stock.")

if __name__ == "__main__":
    main()
