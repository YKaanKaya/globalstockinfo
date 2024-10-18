# app.py

import streamlit as st
from datetime import datetime, timedelta

from data_fetching import (
    get_stock_data,
    get_company_info,
    get_esg_data,
    get_competitors,
    get_news,
    get_analyst_estimates,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
)

from data_processing import (
    compute_returns,
    compute_moving_averages,
    get_rsi,
    get_sentiment_score,
    compute_analyst_consensus,
    generate_recommendation,
)

from visualization import (
    display_stock_chart,
    display_returns_chart,
    display_esg_data,
    display_company_info,
    display_rsi_chart,
    display_recommendation_visualization,
    display_analyst_recommendations,
    display_income_statement,
    display_balance_sheet,
    display_cash_flow,
    create_comparison_chart,
    display_news,
)

from utils import format_large_number

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Enhanced Stock Analysis Dashboard")

    st.sidebar.title("Stock Analysis Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
    period = st.sidebar.selectbox(
        "Select Time Period",
        options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        format_func=lambda x: f"{x[:-1]} {'Month' if x[-1]=='M' else 'Year'}{'s' if x[:-1]!='1' else ''}",
        index=3,
    )

    end_date = datetime.now().strftime('%Y-%m-%d')
    if period.endswith('M'):
        delta_days = int(period[:-1]) * 30
    else:
        delta_days = int(period[:-1]) * 365
    start_date = (datetime.now() - timedelta(days=delta_days)).strftime('%Y-%m-%d')

    st.title(f"{ticker} Enhanced Stock Analysis Dashboard")
    st.write(f"Analyzing data from {start_date} to {end_date}")

    with st.spinner('Fetching data...'):
        stock_data = get_stock_data([ticker], start_date, end_date)
        company_info = get_company_info(ticker)
        esg_data = get_esg_data(ticker)
        news = get_news(ticker)
        sentiment_score = get_sentiment_score(news)
        competitors = get_competitors(ticker)
        comparison_data = None
        if competitors:
            comparison_tickers = [ticker] + competitors
            comparison_stock_data = get_stock_data(comparison_tickers, start_date, end_date)
            if comparison_stock_data is not None:
                adj_close = comparison_stock_data['Adj Close']
                if isinstance(adj_close, pd.Series):
                    adj_close = adj_close.to_frame()
                comparison_data = (adj_close.pct_change() + 1).cumprod()
        income_statement = get_income_statement(ticker)
        balance_sheet = get_balance_sheet(ticker)
        cash_flow = get_cash_flow(ticker)
        analyst_estimates = get_analyst_estimates(ticker)
        analyst_consensus = compute_analyst_consensus(analyst_estimates)

    if stock_data is not None and not stock_data.empty:
        stock_data = stock_data[ticker] if ticker in stock_data else stock_data
        stock_data = compute_returns(stock_data)
        stock_data = compute_moving_averages(stock_data)
        stock_data = get_rsi(stock_data)

        # Generate Recommendation
        recommendation, factors = generate_recommendation(
            ticker, company_info, esg_data, sentiment_score, stock_data, analyst_consensus
        )

        # Update top-level metrics to include Recommendation
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric(
            "Current Price",
            f"${stock_data['Close'].iloc[-1]:.2f}",
            f"{stock_data['Daily Return'].iloc[-1]:.2%}",
        )
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
        col6.metric("Recommendation", recommendation)

        # Create Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📈 Stock Chart",
                "🌿 ESG Analysis",
                "ℹ️ Company Info",
                "📰 News & Sentiment",
                "🔍 Unique Insights",
                "📊 Financials",
            ]
        )

        # Tab 1: Stock Chart
        with tab1:
            st.header("Stock Price Analysis")
            display_stock_chart(stock_data, ticker)
            st.header("Returns Analysis")
            display_returns_chart(stock_data, ticker)
            st.header("Technical Indicators")
            display_rsi_chart(stock_data)

        # Tab 2: ESG Analysis
        with tab2:
            if esg_data is not None:
                display_esg_data(esg_data)
            else:
                st.warning("ESG data not available for this stock.")

        # Tab 3: Company Info
        with tab3:
            if company_info:
                display_company_info(company_info)
            else:
                st.warning("Company information not available.")

        # Tab 4: News & Sentiment
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                if news:
                    display_news(news)
                else:
                    st.warning("No recent news available for this stock.")
            with col2:
                st.subheader("Sentiment Analysis")
                st.write(f"The overall sentiment based on recent news headlines is **{sentiment_score}**.")

        # Tab 5: Unique Insights
        with tab5:
            st.header("Unique Insights")

            st.subheader("Automated Stock Recommendation")
            st.write("**Recommendation:**", recommendation)
            st.write(
                "**Disclaimer:** This recommendation is generated automatically based on predefined criteria and is not financial advice. This app is intended for improving technical skills and sharing them with potential interested parties."
            )

            st.subheader("Factors Considered")
            for factor, assessment in factors.items():
                st.write(f"- **{factor}**: {assessment}")

            # Visualization for Recommendation
            st.subheader("Recommendation Visualization")
            weights = {
                'P/E Ratio': 1,
                'Dividend Yield': 0.5,
                'ESG Score': 0.5,
                'Sentiment': 1,
                'RSI': 1,
                'Analyst Consensus': 2,
            }
            display_recommendation_visualization(recommendation, factors, weights)

            # Competitor Comparison
            if comparison_data is not None and not comparison_data.empty:
                st.subheader("Competitor Comparison")
                comparison_chart = create_comparison_chart(comparison_data)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
                else:
                    st.warning("Competitor comparison data not available.")
            else:
                st.warning("Competitor comparison data not available.")

            # Analyst Recommendations
            st.subheader("Analyst Recommendations")
            display_analyst_recommendations(analyst_consensus)

        # Tab 6: Financials
        with tab6:
            st.header("Financial Statements")
            fin_tab1, fin_tab2, fin_tab3 = st.tabs(
                ["Income Statement", "Balance Sheet", "Cash Flow Statement"]
            )
            with fin_tab1:
                if income_statement is not None:
                    display_income_statement(income_statement)
                else:
                    st.warning("Income statement data not available.")
            with fin_tab2:
                if balance_sheet is not None:
                    display_balance_sheet(balance_sheet)
                else:
                    st.warning("Balance sheet data not available.")
            with fin_tab3:
                if cash_flow is not None:
                    display_cash_flow(cash_flow)
                else:
                    st.warning("Cash flow data not available.")

    else:
        st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

    st.markdown("---")
    st.markdown("Data provided by Yahoo Finance and Alpha Vantage. This dashboard is for informational purposes only.")

if __name__ == "__main__":
    main()
