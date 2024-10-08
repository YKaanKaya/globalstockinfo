import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Function to compute cumulative returns
def compute_cumulative_return(data):
    data['Cumulative Return'] = (1 + data['Adj Close'].pct_change()).cumprod()
    return data

# Function to compute moving averages
def compute_moving_averages(data, windows=[50, 200]):
    for window in windows:
        data[f'MA{window}'] = data['Adj Close'].rolling(window=window).mean()
    return data

# ESG data scraper using Selenium
@st.cache
def get_esg_data_selenium(ticker):
    options = Options()
    options.add_argument("--headless")
    service = Service("/path/to/chromedriver")  # Update this path to the ChromeDriver executable
    driver = webdriver.Chrome(service=service, options=options)
    
    url = f"https://uk.finance.yahoo.com/quote/{ticker}/sustainability?p={ticker}"
    driver.get(url)
    
    time.sleep(5)  # Wait for the page to load

    esg_data = {}

    try:
        total_esg = driver.find_element(By.CSS_SELECTOR, "div.Fz(36px).Fw(600).D(ib).Mend(5px)").text
        esg_data["Total ESG Risk Score"] = float(total_esg)
    except:
        esg_data["Total ESG Risk Score"] = None

    try:
        risk_scores = driver.find_elements(By.CSS_SELECTOR, "div.D(ib).Fz(23px).smartphone_Fz(22px).Fw(600)")
        esg_data["Environmental Risk Score"] = float(risk_scores[0].text)
        esg_data["Social Risk Score"] = float(risk_scores[1].text)
        esg_data["Governance Risk Score"] = float(risk_scores[2].text)
    except:
        esg_data["Environmental Risk Score"] = None
        esg_data["Social Risk Score"] = None
        esg_data["Governance Risk Score"] = None

    try:
        controversy = driver.find_element(By.CSS_SELECTOR, "div.D(ib).Fz(36px).Fw(500)").text
        esg_data["Controversy Level"] = int(controversy)
    except:
        esg_data["Controversy Level"] = None

    driver.quit()
    return esg_data

# Mapping ESG risk score to risk level
def map_esg_risk_to_level(score):
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

# Displaying ESG data in a table
def display_esg_data_table(selected_symbols, esg_data_list):
    esg_df = pd.DataFrame(esg_data_list)
    esg_df.insert(0, 'Ticker', selected_symbols)
    st.markdown("### ESG Data Table:")
    st.table(esg_df)

# Displaying risk levels
def display_risk_levels(tickers, esg_scores):
    st.write("### ESG Risk Levels")
    risk_levels = ["Very Low", "Low", "Medium", "High", "Severe"]
    score_ranges = [5, 15, 25, 35, 45]
    colors = ["#FFEDCC", "#FFDB99", "#FFC266", "#FF9900", "#FF6600"]

    df = pd.DataFrame({
        'Risk Level': risk_levels,
        'Score Range': score_ranges,
        'Color': colors
    })

    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Score Range']],
            y=[row['Risk Level']],
            marker_color=row['Color'],
            orientation='h',
            showlegend=False
        ))

    for ticker, score in zip(tickers, esg_scores):
        score_level = map_esg_risk_to_level(score)
        fig.add_annotation(
            x=score,
            y=score_level,
            text=f"{ticker}: {score}",
            showarrow=False,
            font=dict(color='black', size=12)
        )

    fig.update_layout(
        title="ESG Risk Levels",
        xaxis_title="Score",
        yaxis_title="Risk Level"
    )
    st.plotly_chart(fig)

# Displaying stock price chart with moving averages
def display_stock_price_chart(data, ticker):
    fig = go.Figure()

    # Plotting adjusted close price
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Adj Close'))

    # Plotting moving averages
    for window in [50, 200]:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA{window}'], mode='lines', name=f'MA{window}'))

    fig.update_layout(title=f'Stock Prices and Moving Averages for {ticker}', xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig)

# Main application logic
def main():
    st.title("Financial Data Application")

    st.sidebar.markdown("### Select Stock Tickers")
    selected_tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT").split(',')

    period = st.sidebar.selectbox("Select Time Period:", ["1y", "2y", "5y", "10y"])
    interval = st.sidebar.selectbox("Select Time Interval:", ["1d", "1wk", "1mo"])

    if selected_tickers:
        for ticker in selected_tickers:
            ticker = ticker.strip().upper()
            stock_data = yf.download(ticker, period=period, interval=interval)

            if not stock_data.empty:
                stock_data = compute_cumulative_return(stock_data)
                stock_data = compute_moving_averages(stock_data)
                display_stock_price_chart(stock_data, ticker)

                if st.sidebar.checkbox("Show ESG data"):
                    esg_data = get_esg_data_selenium(ticker)
                    display_esg_data_table([ticker], [esg_data])
                    display_risk_levels([ticker], [esg_data["Total ESG Risk Score"]])

if __name__ == "__main__":
    main()
