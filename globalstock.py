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
    st.title("Stock Data Downloader")

    # Descriptive text for multi-selection
    st.write("Enter stock tickers separated by spaces. Example: AAPL GOOGL MSFT")

    # User input for stock ticker (default to AAPL) and period
    tickers = st.text_input("Enter Stock Tickers:", "AAPL").upper().split()
    available_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    period = st.selectbox("Select Period:", available_periods, index=2)  # default to "1mo"

    # Interval selection
    available_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = st.selectbox("Select Interval:", available_intervals, index=8)  # default to "1d"

    # Options for cumulative return and moving averages
    add_cumulative_return = st.checkbox('Add Cumulative Return')
    add_moving_averages = st.checkbox('Add Moving Averages')

    if st.button("Download"):
        try:
            data_frames = []
            for ticker in tickers:
                data = yf.download(ticker, period=period, interval=interval)
                
                if add_cumulative_return:
                    data = compute_cumulative_return(data)
                
                if add_moving_averages:
                    data = compute_moving_averages(data)

                data_frames.append(data)

            # Concatenate all dataframes for multiple tickers
            if len(data_frames) > 1:
                final_data = pd.concat(data_frames, keys=tickers, names=['Ticker', 'Date'])
            else:
                final_data = data_frames[0]

            st.write(final_data)

            # Allow user to download the data
            csv = final_data.to_csv(index=True)
            b64 = b64encode(csv.encode()).decode()  # B64 encoding for downloading
            filename = "_".join(tickers) + f"_{period}_{interval}.csv"
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    from base64 import b64encode
    main()
