# visualization.py

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import format_large_number

def display_stock_chart(data, ticker):
    """Display stock price chart with moving averages and volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Price'), row=1, col=1)

    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50 Day MA', line=dict(color='orange', width=1)), row=1, col=1)
    if 'MA200' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='200 Day MA', line=dict(color='red', width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=2, col=1)

    fig.update_layout(
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def display_returns_chart(data, ticker):
    """Display cumulative returns chart."""
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
    """Display ESG scores as a bar chart."""
    st.subheader("ESG Data")
    relevant_metrics = ['totalEsg', 'environmentScore', 'socialScore', 'governanceScore']
    numeric_data = esg_data[esg_data.index.isin(relevant_metrics)]

    # Mapping for better metric names
    metric_names = {
        'totalEsg': 'Total ESG Score',
        'environmentScore': 'Environmental Score',
        'socialScore': 'Social Score',
        'governanceScore': 'Governance Score'
    }

    fig = go.Figure()
    colors = {'Total ESG Score': 'purple', 'Environmental Score': 'green', 'Social Score': 'blue', 'Governance Score': 'orange'}

    for metric in numeric_data.index:
        readable_metric = metric_names.get(metric, metric)
        fig.add_trace(go.Bar(
            x=[readable_metric],
            y=[numeric_data.loc[metric].values[0]],
            name=readable_metric,
            marker_color=colors.get(readable_metric, 'gray')
        ))

    fig.update_layout(
        title="ESG Scores",
        yaxis_title="Score",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def display_company_info(info):
    """Display company information with metrics."""
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sector", info.get('sector', 'N/A'))
        if isinstance(info.get('fullTimeEmployees'), (int, float)):
            st.metric("Full Time Employees", f"{int(info.get('fullTimeEmployees')):,}")
        else:
            st.metric("Full Time Employees", "N/A")
    with col2:
        st.metric("Industry", info.get('industry', 'N/A'))
        st.metric("Country", info.get('country', 'N/A'))

    st.subheader("Financial Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", format_large_number(info.get('marketCap')))
    with col2:
        forward_pe = info.get('forwardPE', 'N/A')
        forward_pe_display = f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else 'N/A'
        st.metric("Forward P/E", forward_pe_display)
    with col3:
        dividend_yield = info.get('dividendYield', 'N/A')
        dividend_yield_display = f"{dividend_yield:.2%}" if isinstance(dividend_yield, (int, float)) else 'N/A'
        st.metric("Dividend Yield", dividend_yield_display)

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', 'N/A'))

    st.subheader("Contact Information")
    if info.get('website', 'N/A') != 'N/A':
        st.markdown(f"**Website:** [{info.get('website')}]({info.get('website')})")
    else:
        st.markdown(f"**Website:** N/A")
    st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    address_parts = [info.get('address1', ''), info.get('city', ''), info.get('state', ''), info.get('zip', ''), info.get('country', '')]
    address = ', '.join(part for part in address_parts if part)
    st.markdown(f"**Address:** {address}")

def display_rsi_chart(data):
    """Display RSI chart."""
    if 'RSI' not in data.columns:
        st.warning("RSI data not available.")
        return
    st.subheader("Relative Strength Index (RSI)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name='RSI'
    ))
    fig.update_layout(
        title="RSI Over Time",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def display_recommendation_visualization(recommendation, factors, weights):
    """Visualize recommendation factors and overall score."""
    # Convert factors to numerical scores
    factor_scores = []
    factor_names = []
    for factor, assessment in factors.items():
        if 'Positive' in assessment:
            score = 1 * weights[factor]
        elif 'Negative' in assessment:
            score = -1 * weights[factor]
        else:
            score = 0
        factor_names.append(factor)
        factor_scores.append(score)

    # Create a bar chart of factor scores
    fig = go.Figure(data=[
        go.Bar(x=factor_names, y=factor_scores, marker_color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in factor_scores])
    ])
    fig.update_layout(
        title="Factor Scores",
        xaxis_title="Factors",
        yaxis_title="Weighted Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def display_analyst_recommendations(consensus):
    """Display analyst recommendations as a pie chart."""
    if consensus is None:
        st.warning("No analyst consensus available.")
        return
    labels = list(consensus.keys())
    values = list(consensus.values())
    if sum(values) == 0:
        st.warning("No analyst recommendations data to display.")
        return
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Analyst Recommendations Consensus")
    st.plotly_chart(fig, use_container_width=True)

# visualization.py

def display_income_statement(income_statement):
    """Display income statement data."""
    st.subheader("Income Statement")
    if income_statement is None or income_statement.empty:
        st.warning("Income statement data not available.")
        return

    # Set index to the fiscal date
    if 'fiscalDateEnding' in income_statement.columns:
        reports = income_statement.set_index('fiscalDateEnding')
    elif 'endDate' in income_statement.columns:
        reports = income_statement.set_index('endDate')
    else:
        st.warning("Fiscal date information not available.")
        return

    # Convert columns to lowercase for consistency
    reports.columns = reports.columns.str.lower()

    # Define possible column names from both data sources
    total_revenue_cols = ['totalrevenue', 'total revenue', 'revenue']
    gross_profit_cols = ['grossprofit', 'gross profit']
    net_income_cols = ['netincome', 'net income']

    # Map the columns
    columns_to_display = {}
    for possible_names, display_name in zip(
        [total_revenue_cols, gross_profit_cols, net_income_cols],
        ['Total Revenue', 'Gross Profit', 'Net Income']
    ):
        for col_name in possible_names:
            if col_name in reports.columns:
                columns_to_display[display_name] = col_name
                break
        else:
            st.warning(f"{display_name} not found in the income statement data.")
            columns_to_display[display_name] = None

    # Remove any None values
    columns_to_display = {k: v for k, v in columns_to_display.items() if v is not None}

    if not columns_to_display:
        st.warning("Required columns not found in the income statement data.")
        return

    # Select and rename columns
    reports_selected = reports[list(columns_to_display.values())]
    reports_selected.rename(columns={v: k for k, v in columns_to_display.items()}, inplace=True)

    # Convert to numeric
    reports_selected = reports_selected.apply(pd.to_numeric, errors='coerce')

    # Transpose for display
    reports_transposed = reports_selected.transpose()

    # Format and display
    formatted_reports = reports_transposed.style.format("{:,.0f}")
    st.dataframe(formatted_reports)


def display_balance_sheet(balance_sheet):
    """Display balance sheet data."""
    st.subheader("Balance Sheet")
    if balance_sheet is None or balance_sheet.empty:
        st.warning("Balance sheet data not available.")
        return

    # Set index to the fiscal date
    if 'fiscalDateEnding' in balance_sheet.columns:
        reports = balance_sheet.set_index('fiscalDateEnding')
    elif 'endDate' in balance_sheet.columns:
        reports = balance_sheet.set_index('endDate')
    else:
        st.warning("Fiscal date information not available.")
        return

    # Convert columns to lowercase
    reports.columns = reports.columns.str.lower()

    # Define possible column names
    total_assets_cols = ['totalassets', 'total assets']
    total_liabilities_cols = ['totalliabilities', 'total liabilities']
    total_equity_cols = ['totalequity', 'total stockholder equity', 'total shareholder equity']

    # Map the columns
    columns_to_display = {}
    for possible_names, display_name in zip(
        [total_assets_cols, total_liabilities_cols, total_equity_cols],
        ['Total Assets', 'Total Liabilities', 'Total Shareholder Equity']
    ):
        for col_name in possible_names:
            if col_name in reports.columns:
                columns_to_display[display_name] = col_name
                break
        else:
            st.warning(f"{display_name} not found in the balance sheet data.")
            columns_to_display[display_name] = None

    # Remove any None values
    columns_to_display = {k: v for k, v in columns_to_display.items() if v is not None}

    if not columns_to_display:
        st.warning("Required columns not found in the balance sheet data.")
        return

    # Select and rename columns
    reports_selected = reports[list(columns_to_display.values())]
    reports_selected.rename(columns={v: k for k, v in columns_to_display.items()}, inplace=True)

    # Convert to numeric
    reports_selected = reports_selected.apply(pd.to_numeric, errors='coerce')

    # Transpose for display
    reports_transposed = reports_selected.transpose()

    # Format and display
    formatted_reports = reports_transposed.style.format("{:,.0f}")
    st.dataframe(formatted_reports)


def display_cash_flow(cash_flow):
    """Display cash flow statement data."""
    st.subheader("Cash Flow Statement")
    if cash_flow is None or cash_flow.empty:
        st.warning("Cash flow data not available.")
        return

    # Set index to the fiscal date
    if 'fiscalDateEnding' in cash_flow.columns:
        reports = cash_flow.set_index('fiscalDateEnding')
    elif 'endDate' in cash_flow.columns:
        reports = cash_flow.set_index('endDate')
    else:
        st.warning("Fiscal date information not available.")
        return

    # Convert columns to lowercase
    reports.columns = reports.columns.str.lower()

    # Define possible column names
    operating_cashflow_cols = ['operatingcashflow', 'total cash from operating activities']
    investing_cashflow_cols = ['cashflowfrominvestment', 'total cashflows from investing activities']
    financing_cashflow_cols = ['cashflowfromfinancing', 'total cash from financing activities']
    net_income_cols = ['netincome', 'net income']

    # Map the columns
    columns_to_display = {}
    for possible_names, display_name in zip(
        [operating_cashflow_cols, investing_cashflow_cols, financing_cashflow_cols, net_income_cols],
        ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Net Income']
    ):
        for col_name in possible_names:
            if col_name in reports.columns:
                columns_to_display[display_name] = col_name
                break
        else:
            st.warning(f"{display_name} not found in the cash flow data.")
            columns_to_display[display_name] = None

    # Remove any None values
    columns_to_display = {k: v for k, v in columns_to_display.items() if v is not None}

    if not columns_to_display:
        st.warning("Required columns not found in the cash flow data.")
        return

    # Select and rename columns
    reports_selected = reports[list(columns_to_display.values())]
    reports_selected.rename(columns={v: k for k, v in columns_to_display.items()}, inplace=True)

    # Convert to numeric
    reports_selected = reports_selected.apply(pd.to_numeric, errors='coerce')

    # Transpose for display
    reports_transposed = reports_selected.transpose()

    # Format and display
    formatted_reports = reports_transposed.style.format("{:,.0f}")
    st.dataframe(formatted_reports)


def create_comparison_chart(comparison_data):
    """Create a Plotly chart comparing cumulative returns."""
    if comparison_data is None or comparison_data.empty:
        st.warning("No data available for comparison.")
        return None

    fig = go.Figure()
    for column in comparison_data.columns:
        fig.add_trace(go.Scatter(x=comparison_data.index, y=comparison_data[column], mode='lines', name=column))
    fig.update_layout(title="1 Year Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
    return fig

def display_news(news):
    """Display the latest news articles."""
    st.subheader("Latest News")
    if not news:
        st.warning("No news available.")
        return
    for article in news[:5]:  # Display top 5 news articles
        st.markdown(f"### [{article['title']}]({article['link']})")
        try:
            pub_time = datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pub_time = "N/A"
        st.markdown(f"*Published on: {pub_time}*")
        st.markdown("---")
