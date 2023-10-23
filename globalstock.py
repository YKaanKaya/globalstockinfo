import yfinance as yf
import streamlit as st
import pandas as pd

def compute_cumulative_return(data):
    data['Cumulative Return'] = (1 + data['Adj Close'].pct_change()).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    for window in windows:
        data[f'MA{window}'] = data['Adj Close'].rolling(window=window).mean()
    return data

def main():
    st.title("Stock Data Downloader 📈")

    st.sidebar.header("Settings")

    # Ticker input
    tickers = st.sidebar.text_input("Enter Stock Tickers:", "AAPL").upper().split()
    st.sidebar.text("Example: AAPL GOOGL MSFT")

    # Period selection
    available_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.sidebar.selectbox("Select Period:", available_periods, index=2)

    # Interval selection
    available_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = st.sidebar.selectbox("Select Interval:", available_intervals, index=8)

    # Additional data features
    st.sidebar.subheader("Additional Data Features")
    add_cumulative_return = st.sidebar.checkbox('Add Cumulative Return')
    add_moving_averages = st.sidebar.checkbox('Add Moving Averages')

    st.sidebar.text("Note: Data is fetched from Yahoo Finance.")
    st.sidebar.text("© 2023 Your Company Name")

    if st.sidebar.button("Download Data"):
        with st.spinner("Fetching data..."):
            try:
                data_frames = []
                for ticker in tickers:
                    data = yf.download(ticker, period=period, interval=interval)
                    
                    if add_cumulative_return:
                        data = compute_cumulative_return(data)
                    
                    if add_moving_averages:
                        data = compute_moving_averages(data)

                    data_frames.append(data)

                if len(data_frames) > 1:
                    final_data = pd.concat(data_frames, keys=tickers, names=['Ticker', 'Date'])
                else:
                    final_data = data_frames[0]

                st.dataframe(final_data.style.highlight_max(axis=0))

                # Allow user to download the data
                csv = final_data.to_csv(index=True)
                b64 = b64encode(csv.encode()).decode()
                filename = "_".join(tickers) + f"_{period}_{interval}.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    from base64 import b64encode
    main()
