import yfinance as yf
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import ticker_fetcher
from base64 import b64encode

# Utility functions
def compute_cumulative_return(data):
    data['Cumulative Return'] = (1 + data['Adj Close'].pct_change()).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    for window in windows:
        data[f'MA{window}'] = data['Adj Close'].rolling(window=window).mean()
    return data

# Scrape the ESG data
@st.cache_data
def get_esg_data_with_headers_and_error_handling(ticker):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    url = f"https://uk.finance.yahoo.com/quote/{ticker}/sustainability?p={ticker}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    result = {}

    try:
        total_esg_risk_score = soup.find("div", {"class": "Fz(36px) Fw(600) D(ib) Mend(5px)"}).text
        result["Total ESG risk score"] = float(total_esg_risk_score)
    except:
        result["Total ESG risk score"] = None

    scores = soup.find_all("div", {"class": "D(ib) Fz(23px) smartphone_Fz(22px) Fw(600)"})
    try:
        result["Environment risk score"] = float(scores[0].text)
    except:
        result["Environment risk score"] = None

    try:
        result["Social risk score"] = float(scores[1].text)
    except:
        result["Social risk score"] = None

    try:
        result["Governance risk score"] = float(scores[2].text)
    except:
        result["Governance risk score"] = None

    try:
        controversy_level = soup.find("div", {"class": "D(ib) Fz(36px) Fw(500)"}).text
        result["Controversy level"] = int(controversy_level)
    except:
        result["Controversy level"] = None

    return result

# Function to map ESG risk score to risk level
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

# Display functions
def display_esg_data_table(selected_symbols, esg_data_list):
    esg_df = pd.DataFrame(esg_data_list)
    esg_df.insert(0, 'Ticker', selected_symbols)
    st.markdown("### ESG Data Table:")
    st.write("Environmental, Social, and Governance (ESG) metrics evaluate a company's commitment to sustainability and ethical practices. Lower risk scores typically indicate better corporate responsibility.")
    st.table(esg_df)

def display_risk_levels(tickers, esg_scores):
    st.write("### ESG Risk Levels:")
    st.write("Visualize how the selected tickers rank in terms of ESG risk. This can give insights into potential sustainability challenges the company may face, lower better.")
    risk_levels = ["Very Low", "Low", "Medium", "High", "Severe"]
    score_ranges = [5, 15, 25, 35, 45]
    colors = ["#FFEDCC", "#FFDB99", "#FFC266", "#FF9900", "#FF6600"]

    df = pd.DataFrame({
        'Risk Level': risk_levels,
        'Score Range': score_ranges,
        'Color': colors
    })

    fig = px.bar(df, x='Score Range', y='Risk Level', color='Color', orientation='h',
                 color_discrete_map=dict(zip(df['Color'], df['Color'])))

    for ticker, score in zip(tickers, esg_scores):
        score_position = risk_levels.index(map_esg_risk_to_level(score))
        annotation_x = df.loc[score_position, 'Score Range'] - 3
        fig.add_annotation(
            x=annotation_x,
            y=risk_levels[score_position],
            text=f"{ticker}: {score}",
            showarrow=False,
            font=dict(color='black', size=12),
            xshift=10
        )

    fig.update_layout(
        title="ESG Risk Levels and Ticker Scores",
        xaxis_title="Score Range",
        yaxis_title="Risk Level",
        showlegend=False,
        plot_bgcolor='#2e2e2e',
        paper_bgcolor='#2e2e2e',
        font=dict(color='white')
    )

    st.plotly_chart(fig)

def display_stock_price_chart(data, ticker):
    fig = go.Figure()

    # Plotting adjusted close price
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Adj Close'))
    
    # Plotting moving averages
    for window in [50, 200]:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA{window}'], mode='lines', name=f'MA{window}'))
    
    fig.update_layout(title=f'Stock Prices and Moving Averages for {ticker}', xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig)

def display_esg_score_progress_bar(ticker, score):
    st.write(f"### ESG Score for {ticker}")    
    max_score = 50  # assuming max possible score is 50
    progress_bar = st.progress(score / max_score)
    if score >= 40:
        progress_bar.color = 'red'
    elif score >= 30:
        progress_bar.color = 'orange'
    elif score >= 20:
        progress_bar.color = 'yellow'
    else:
        progress_bar.color = 'green'
    
    st.write(f"ESG Risk Score: {score}")

def main():
    st.title("Financial Data Application")
    st.markdown("Welcome to the Financial Data Application. This platform enables you to fetch stock price data for publicly traded companies, visualize key metrics, and delve into ESG (Environmental, Social, and Governance) data. Whether you're a professional investor, a student, or just curious about the stock market, this tool is designed to be both informative and intuitive.")
    st.markdown("With this foundation, feel free to explore the various features of the application. Happy investing!")
   
    st.sidebar.markdown("### Stock Ticker Selection")    
    default_tickers = ["NVDA"]
    
    common_tickers = ticker_fetcher.get_tickers()

    # Ensure default_tickers are in common_tickers
    default_tickers = [ticker for ticker in default_tickers if ticker in common_tickers]
    if not default_tickers:
        default_tickers = [common_tickers[0]]

    st.sidebar.markdown("**Select one or more stock tickers from the predefined list below.**")
    st.sidebar.markdown("_These represent the stock symbols for publicly traded companies._")
    selected_from_predefined = st.sidebar.multiselect("Select Tickers from List:", common_tickers, default=default_tickers)

    st.sidebar.markdown("**If you can't find your desired stock ticker in the list, or prefer to manually enter them, you can type them below.**")
    st.sidebar.markdown("_Separate multiple tickers with commas._")
    custom_tickers_input = st.sidebar.text_input("Or enter custom tickers (comma separated):")
    custom_tickers = [ticker.strip().upper() for ticker in custom_tickers_input.split(',') if ticker.strip()]

    selected_tickers = list(set(selected_from_predefined + custom_tickers))

    st.sidebar.markdown("### Time Settings")
    st.sidebar.markdown("Set the range (`Time Period`) and granularity (`Time Interval`) of the stock data. For example, a '1y' period with a '1mo' interval shows monthly data points over a year.")
    period = st.sidebar.selectbox("Select Time Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    interval = st.sidebar.selectbox("Select Time Interval:", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])

    st.sidebar.markdown("### ESG Data")    
    display_esg = st.sidebar.checkbox("Display ESG data", True)   
    display_esg_risk_levels = st.sidebar.checkbox("Display ESG risk levels", True)
    
    st.sidebar.markdown("### Data Export")
    download_link = st.sidebar.button("Download Data as CSV")
   
    st.sidebar.markdown("### Refresh Data")
    refresh_data = st.sidebar.button("Refresh Data")
    
    data_dict = {}
    esg_data_list = []

    if refresh_data or (not selected_tickers):
        selected_tickers = default_tickers

    with st.spinner("Fetching data..."):
        for ticker in selected_tickers:
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                st.error(f"No data available for {ticker} in the selected date range.")
                continue

            data['Symbol'] = ticker
            data = compute_cumulative_return(data)
            data = compute_moving_averages(data)
            data_dict[ticker] = data

            if display_esg or display_esg_risk_levels:
                esg_data = get_esg_data_with_headers_and_error_handling(ticker)
                esg_data_list.append(esg_data)

    st.write("### Stock Data Overview")
    final_data = pd.concat(data_dict.values())
    st.dataframe(final_data)

    st.markdown("### Moving Averages")
    st.markdown("A **moving average** smoothens price data to create a single flowing line, which makes it easier to identify the direction of the trend. The two commonly used moving averages displayed here are:")
    st.markdown("- **50-day moving average (MA50)**: This is the average of the closing prices over the last 50 days. It's a shorter-term moving average, and its crossover with longer-term MAs can signal potential buy or sell opportunities.")
    st.markdown("- **200-day moving average (MA200)**: Represents the average of the closing prices over the last 200 days. It's used to gauge the overall trend of the stock. Stocks trading above their 200-day moving average are often considered in an uptrend, and those below are considered in a downtrend.")
    
    st.markdown("### Cumulative Returns")
    st.markdown("The **cumulative return** is the entire amount of money an investment has made for an investor, irrespective of time. It's a raw measurement on how well (or poorly) the investment did. Here's how to interpret it:")
    st.markdown("- A cumulative return of 1.0 indicates that the investment's value hasn't changed.")
    st.markdown("- A number above 1.0 indicates a profit. For example, 1.5 means the investment has returned 150% of its initial value.")
    st.markdown("- A number less than 1.0 indicates a loss. For example, 0.8 means the investment has returned only 80% of its initial value, representing a 20% loss.")

    for ticker in selected_tickers:
        if ticker in data_dict:
            display_stock_price_chart(data_dict[ticker], ticker)
            if display_esg:
                esg_data = get_esg_data_with_headers_and_error_handling(ticker)
                if esg_data["Total ESG risk score"] is not None:
                    display_esg_score_progress_bar(ticker, esg_data["Total ESG risk score"])

    if display_esg:
        display_esg_data_table(selected_tickers, esg_data_list)

    if display_esg_risk_levels:
        esg_scores = [data["Total ESG risk score"] for data in esg_data_list]
        display_risk_levels(selected_tickers, esg_scores)

    if download_link:
        try:
            csv = final_data.to_csv(index=False)
            b64 = b64encode(csv.encode()).decode()
            st.markdown(f"### Download Data as CSV:\n[Download Link](data:file/csv;base64,{b64})")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
