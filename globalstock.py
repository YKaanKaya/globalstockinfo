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
    
# Scrape the ESG data
@st.cache
def get_esg_data_with_headers_and_error_handling(ticker):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
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

def display_esg_data_table(selected_symbols, esg_data_list):
    # Convert the list of dictionaries to a DataFrame
    esg_df = pd.DataFrame(esg_data_list)
    esg_df.insert(0, 'Ticker', selected_symbols)  # Add a column for tickers at the beginning
    st.write("### ESG Data Table:")
    st.table(esg_df)
        
# Function to display ESG risk levels
def display_risk_levels(tickers, esg_scores):
    st.write("### ESG Risk Levels:")
    
    risk_levels = ["Very Low", "Low", "Medium", "High", "Severe"]
    score_ranges = [5, 15, 25, 35, 45]  # Midpoint of each score range for plotting
    colors = ["#FFEDCC", "#FFDB99", "#FFC266", "#FF9900", "#FF6600"]  # Shades of orange from light to dark
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Risk Level': risk_levels,
        'Score Range': score_ranges,
        'Color': colors
    })
    
    # Create a bar chart
    fig = px.bar(df, x='Score Range', y='Risk Level', color='Color', orientation='h',
                 color_discrete_map=dict(zip(df['Color'], df['Color'])))
    
    # Annotate the chart with the scores of all selected tickers
    for ticker, score in zip(tickers, esg_scores):
        score_position = risk_levels.index(map_esg_risk_to_level(score))
        annotation_x = df.loc[score_position, 'Score Range'] - 3  # Adjusted for visibility
        fig.add_annotation(
            x=annotation_x,
            y=risk_levels[score_position],
            text=f"{ticker}: {score}",
            showarrow=False,
            font=dict(color='black', size=12),
            xshift=10
        )
    
    # Update chart aesthetics
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

def display_time_series_chart(symbol_data, selected_symbols, start_date, end_date):
    try:
        filtered_data = symbol_data[
            (symbol_data['Symbol'].isin(selected_symbols)) &
            (symbol_data['Datetime'].dt.date >= start_date) &
            (symbol_data['Datetime'].dt.date <= end_date)
        ]

        if filtered_data.empty:
            st.error("No data available for the selected symbols in the selected date range.")
        else:
            selected_tickers = ', '.join(selected_symbols)  # Join selected tickers with commas
            
            # Create a Plotly line chart
            fig = go.Figure()  # Create a new Plotly figure
            
            # Customize the chart with explicit light colors
            light_colors = ['#FF5733', '#FFBD33', '#33FF57', '#339CFF', '#FF33D1']  # Light colors
            color_mapping = {symbol: light_colors[i % len(light_colors)] for i, symbol in enumerate(selected_symbols)}
            
            for symbol in selected_symbols:
                symbol_data = filtered_data[filtered_data['Symbol'] == symbol]
                # Add trace only once for each unique symbol
                if symbol not in fig.data:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data['Datetime'],
                            y=symbol_data['High'],
                            mode='lines',
                            name=symbol,
                            line=dict(color=color_mapping[symbol], width=2)
                        )
                    )
                
                # Add annotations for highest and lowest trading prices
                min_return_row = symbol_data.loc[symbol_data['Low'].idxmin()]  # Get the row with the minimum 'Low' value
                max_return_row = symbol_data.loc[symbol_data['High'].idxmax()]  # Get the row with the maximum 'High' value
                
                fig.add_annotation(
                    x=min_return_row['Datetime'],
                    y=min_return_row['Low'],
                    text=f"Lowest: ${min_return_row['Low']:.2f}",
                    showarrow=False,
                    arrowhead=4,
                    ax=0,
                    ay=-40
                )
                
                fig.add_annotation(
                    x=max_return_row['Datetime'],
                    y=max_return_row['High'],
                    text=f"Highest: ${max_return_row['High']:.2f}",
                    showarrow=False,
                    arrowhead=4,
                    ax=0,
                    ay=40
                )
            
            # Set chart title
            fig.update_layout(
                title=f"Time Series Chart for {selected_tickers} Tickers",
                xaxis_title="Date",
                yaxis_title="Highest Price"
            )
            
            # Show the chart
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
def main():
    st.title("Stock Data Downloader & ESG Analyzer 📈")

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
    add_esg_scores = st.sidebar.checkbox('Add ESG Scores and Risk Levels')

    st.sidebar.text("Note: Data is fetched from Yahoo Finance.")
    st.sidebar.text("© 2023 Your Company Name")

    if st.sidebar.button("Download Data"):
        with st.spinner("Fetching data..."):
            try:
                data_frames = []
                esg_data_list = []
                for ticker in tickers:
                    data = yf.download(ticker, period=period, interval=interval)
                    
                    if add_cumulative_return:
                        data = compute_cumulative_return(data)
                    
                    if add_moving_averages:
                        data = compute_moving_averages(data)

                    data_frames.append(data)

                    if add_esg_scores:
                        esg_data = get_esg_data_with_headers_and_error_handling(ticker)
                        esg_data_list.append(esg_data)

                if len(data_frames) > 1:
                    final_data = pd.concat(data_frames, keys=tickers, names=['Ticker', 'Date'])
                else:
                    final_data = data_frames[0]

                st.dataframe(final_data.style.highlight_max(axis=0))

                if add_esg_scores and esg_data_list:
                    display_esg_data_table(tickers, esg_data_list)
                    esg_scores = [data["Total ESG risk score"] for data in esg_data_list if data["Total ESG risk score"] is not None]
                    display_risk_levels(tickers, esg_scores)

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

    main()
