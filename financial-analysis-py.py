import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from data_fetcher import format_large_number

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_company_info(info):
    """
    Display company information in an organized layout.
    
    Args:
        info (dict): Dictionary containing company information
    """
    try:
        st.subheader("Company Information")
        
        # Create two columns for basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Industry Classification")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            
            # Company logo if available
            if info.get('logo_url') and info.get('logo_url') != 'N/A':
                st.image(info.get('logo_url'), width=100)
        
        with col2:
            st.markdown("#### Company Size")
            st.markdown(f"**Market Cap:** {format_large_number(info.get('marketCap', 'N/A'))}")
            st.markdown(f"**Employees:** {int(info.get('fullTimeEmployees', 0)):,}" if info.get('fullTimeEmployees', 'N/A') != 'N/A' else "**Employees:** N/A")
            st.markdown(f"**Country:** {info.get('country', 'N/A')}")
        
        # Create three columns for key financial metrics
        st.markdown("#### Key Financial Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Forward P/E", 
                f"{info.get('forwardPE', 'N/A'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A'
            )
            st.metric(
                "Price to Book", 
                f"{info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else 'N/A'
            )
        
        with col2:
            st.metric(
                "Dividend Yield", 
                f"{info.get('dividendYield', 'N/A'):.2%}" if isinstance(info.get('dividendYield'), (int, float)) else 'N/A'
            )
            st.metric(
                "Beta", 
                f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else 'N/A'
            )
        
        with col3:
            st.metric(
                "52W High", 
                f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else 'N/A'
            )
            st.metric(
                "52W Low", 
                f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else 'N/A'
            )
        
        # Company description
        st.markdown("#### Business Overview")
        st.markdown(info.get('longBusinessSummary', 'No description available.'))
        
        # Contact information
        with st.expander("Contact Information"):
            st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
            
            # Format address
            address_parts = [
                info.get('address1', ''), 
                info.get('city', ''), 
                info.get('state', ''), 
                info.get('zip', ''),
                info.get('country', '')
            ]
            address = ', '.join(part for part in address_parts if part)
            
            st.markdown(f"**Address:** {address}")
            st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    
    except Exception as e:
        logger.error(f"Error displaying company info: {str(e)}")
        st.error(f"Error displaying company information: {str(e)}")

def display_income_statement(income_statement):
    """
    Display income statement data in a formatted table with visualizations.
    
    Args:
        income_statement (pd.DataFrame): Income statement data
    """
    try:
        st.subheader("Income Statement")
        
        # Check if income_statement is valid
        if income_statement is None or income_statement.empty:
            st.warning("Income statement data not available.")
            return
        
        # Display the last 5 annual reports
        reports = income_statement.head(5)
        reports = reports.set_index('fiscalDateEnding')
        
        # Select relevant columns with proper names
        column_mapping = {
            'totalRevenue': 'Revenue',
            'costOfRevenue': 'Cost of Revenue',
            'grossProfit': 'Gross Profit',
            'operatingExpenses': 'Operating Expenses',
            'operatingIncome': 'Operating Income',
            'ebit': 'EBIT',
            'interestExpense': 'Interest Expense',
            'incomeBeforeTax': 'Income Before Tax',
            'incomeTaxExpense': 'Income Tax Expense',
            'netIncome': 'Net Income',
            'ebitda': 'EBITDA'
        }
        
        # Select available columns from mapping
        available_columns = [col for col in column_mapping.keys() if col in reports.columns]
        
        if not available_columns:
            st.warning("No relevant income statement data available.")
            return
            
        selected_columns = available_columns
        
        # Rename columns for display
        display_columns = {col: column_mapping[col] for col in selected_columns}
        reports_display = reports[selected_columns].rename(columns=display_columns)
        
        # Convert columns to numeric, coercing errors
        reports_display = reports_display.apply(pd.to_numeric, errors='coerce')
        
        # Sort index in reverse chronological order
        reports_display = reports_display.sort_index(ascending=False)
        
        # Format the index for display
        reports_display.index = reports_display.index.strftime('%Y-%m-%d')
        
        # Display the income statement
        st.dataframe(reports_display.style.format("{:,.0f}"))
        
        # Create a visualization of key metrics
        st.markdown("#### Key Income Statement Metrics")
        
        # Select metrics for visualization
        viz_metrics = ['Revenue', 'Gross Profit', 'Net Income']
        viz_data = reports_display[[col for col in viz_metrics if col in reports_display.columns]]
        
        if not viz_data.empty:
            # Create a bar chart
            fig = go.Figure()
            
            # Add traces for each year
            for year in viz_data.index:
                fig.add_trace(go.Bar(
                    name=year,
                    x=viz_data.columns,
                    y=viz_data.loc[year],
                    text=[f"${val:,.0f}" for val in viz_data.loc[year]],
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title="Income Statement Trends",
                xaxis_title="Metric",
                yaxis_title="Amount ($)",
                barmode='group',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
            # Calculate and display YoY growth rates
            if len(viz_data) > 1:
                st.markdown("#### Year-over-Year Growth")
                
                # Calculate growth rates for each metric
                growth_data = viz_data.pct_change(-1)  # Negative index to calculate YoY correctly for descending years
                
                # Create a heatmap for growth rates
                fig = go.Figure(data=go.Heatmap(
                    z=growth_data.values * 100,
                    x=growth_data.columns,
                    y=growth_data.index,
                    colorscale='RdYlGn',
                    text=[[f"{val:.2%}" for val in row] for row in growth_data.values],
                    texttemplate="%{text}",
                    colorbar=dict(title="Growth %")
                ))
                
                fig.update_layout(
                    title="Year-over-Year Growth Rates",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        logger.error(f"Error displaying income statement: {str(e)}")
        st.error(f"Error displaying income statement: {str(e)}")

def display_balance_sheet(balance_sheet):
    """
    Display balance sheet data in a formatted table with visualizations.
    
    Args:
        balance_sheet (pd.DataFrame): Balance sheet data
    """
    try:
        st.subheader("Balance Sheet")
        
        # Check if balance_sheet is valid
        if balance_sheet is None or balance_sheet.empty:
            st.warning("Balance sheet data not available.")
            return
        
        # Display the last 5 annual reports
        reports = balance_sheet.head(5)
        reports = reports.set_index('fiscalDateEnding')
        
        # Define important columns with proper names
        column_mapping = {
            'totalAssets': 'Total Assets',
            'totalCurrentAssets': 'Current Assets',
            'cashAndCashEquivalentsAtCarryingValue': 'Cash & Equivalents',
            'inventory': 'Inventory',
            'totalNonCurrentAssets': 'Non-Current Assets',
            'propertyPlantEquipment': 'PP&E',
            'goodwill': 'Goodwill',
            'totalLiabilities': 'Total Liabilities',
            'totalCurrentLiabilities': 'Current Liabilities',
            'totalNonCurrentLiabilities': 'Non-Current Liabilities',
            'longTermDebt': 'Long-Term Debt',
            'totalShareholderEquity': 'Shareholder Equity',
            'retainedEarnings': 'Retained Earnings',
            'commonStock': 'Common Stock'
        }
        
        # Select available columns from mapping
        available_columns = [col for col in column_mapping.keys() if col in reports.columns]
        
        if not available_columns:
            st.warning("No relevant balance sheet data available.")
            return
            
        selected_columns = available_columns
        
        # Rename columns for display
        display_columns = {col: column_mapping[col] for col in selected_columns}
        reports_display = reports[selected_columns].rename(columns=display_columns)
        
        # Convert columns to numeric, coercing errors
        reports_display = reports_display.apply(pd.to_numeric, errors='coerce')
        
        # Sort index in reverse chronological order
        reports_display = reports_display.sort_index(ascending=False)
        
        # Format the index for display
        reports_display.index = reports_display.index.strftime('%Y-%m-%d')
        
        # Display the balance sheet
        st.dataframe(reports_display.style.format("{:,.0f}"))
        
        # Visualize key balance sheet metrics
        st.markdown("#### Assets and Liabilities")
        
        # Select key metrics for visualization
        viz_metrics = ['Total Assets', 'Total Liabilities', 'Shareholder Equity']
        viz_data = reports_display[[col for col in viz_metrics if col in reports_display.columns]]
        
        if not viz_data.empty:
            # Create a bar chart
            fig = go.Figure()
            
            # Add traces for each year
            for year in viz_data.index:
                fig.add_trace(go.Bar(
                    name=year,
                    x=viz_data.columns,
                    y=viz_data.loc[year],
                    text=[f"${val:,.0f}" for val in viz_data.loc[year]],
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title="Balance Sheet Composition",
                xaxis_title="Category",
                yaxis_title="Amount ($)",
                barmode='group',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate financial ratios
            if 'Total Assets' in reports_display.columns and 'Total Liabilities' in reports_display.columns:
                st.markdown("#### Key Financial Ratios")
                
                col1, col2, col3 = st.columns(3)
                
                # Calculate debt-to-equity ratio if possible
                if 'Total Liabilities' in reports_display.columns and 'Shareholder Equity' in reports_display.columns:
                    debt_equity = reports_display['Total Liabilities'] / reports_display['Shareholder Equity']
                    with col1:
                        st.metric("Debt-to-Equity", f"{debt_equity.iloc[0]:.2f}")
                
                # Calculate debt-to-assets ratio
                if 'Total Liabilities' in reports_display.columns and 'Total Assets' in reports_display.columns:
                    debt_assets = reports_display['Total Liabilities'] / reports_display['Total Assets']
                    with col2:
                        st.metric("Debt-to-Assets", f"{debt_assets.iloc[0]:.2f}")
                
                # Calculate current ratio if possible
                if 'Current Assets' in reports_display.columns and 'Current Liabilities' in reports_display.columns:
                    current_ratio = reports_display['Current Assets'] / reports_display['Current Liabilities']
                    with col3:
                        st.metric("Current Ratio", f"{current_ratio.iloc[0]:.2f}")
        
    except Exception as e:
        logger.error(f"Error displaying balance sheet: {str(e)}")
        st.error(f"Error displaying balance sheet: {str(e)}")

def display_cash_flow(cash_flow):
    """
    Display cash flow statement data in a formatted table with visualizations.
    
    Args:
        cash_flow (pd.DataFrame): Cash flow statement data
    """
    try:
        st.subheader("Cash Flow Statement")
        
        # Check if cash_flow is valid
        if cash_flow is None or cash_flow.empty:
            st.warning("Cash flow data not available.")
            return
        
        # Display the last 5 annual reports
        reports = cash_flow.head(5)
        reports = reports.set_index('fiscalDateEnding')
        
        # Define important columns with proper names
        column_mapping = {
            'operatingCashflow': 'Operating Cash Flow',
            'cashflowFromInvestment': 'Investing Cash Flow',
            'cashflowFromFinancing': 'Financing Cash Flow',
            'netIncome': 'Net Income',
            'capitalExpenditures': 'Capital Expenditures',
            'dividendPayout': 'Dividends',
            'freeCashFlow': 'Free Cash Flow',
            'changeInCashAndCashEquivalents': 'Change in Cash'
        }
        
        # Select available columns from mapping
        available_columns = [col for col in column_mapping.keys() if col in reports.columns]
        
        if not available_columns:
            st.warning("No relevant cash flow data available.")
            return
            
        selected_columns = available_columns
        
        # Rename columns for display
        display_columns = {col: column_mapping[col] for col in selected_columns}
        reports_display = reports[selected_columns].rename(columns=display_columns)
        
        # Convert columns to numeric, coercing errors
        reports_display = reports_display.apply(pd.to_numeric, errors='coerce')
        
        # Sort index in reverse chronological order
        reports_display = reports_display.sort_index(ascending=False)
        
        # Format the index for display
        reports_display.index = reports_display.index.strftime('%Y-%m-%d')
        
        # Display the cash flow statement
        st.dataframe(reports_display.style.format("{:,.0f}"))
        
        # Visualize cash flow components
        st.markdown("#### Cash Flow Components")
        
        # Select key metrics for visualization
        viz_metrics = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']
        viz_data = reports_display[[col for col in viz_metrics if col in reports_display.columns]]
        
        if not viz_data.empty:
            # Create a bar chart
            fig = go.Figure()
            
            # Add traces for each metric
            for column in viz_data.columns:
                fig.add_trace(go.Bar(
                    name=column,
                    x=viz_data.index,
                    y=viz_data[column],
                    text=[f"${val:,.0f}" for val in viz_data[column]],
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title="Cash Flow Trends",
                xaxis_title="Year",
                yaxis_title="Amount ($)",
                barmode='group',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display free cash flow analysis if available
            if 'Free Cash Flow' in reports_display.columns or ('Operating Cash Flow' in reports_display.columns and 'Capital Expenditures' in reports_display.columns):
                st.markdown("#### Free Cash Flow Analysis")
                
                # Calculate FCF if not already provided
                if 'Free Cash Flow' not in reports_display.columns and 'Operating Cash Flow' in reports_display.columns and 'Capital Expenditures' in reports_display.columns:
                    reports_display['Free Cash Flow'] = reports_display['Operating Cash Flow'] - reports_display['Capital Expenditures']
                
                # Create a line chart for FCF
                if 'Free Cash Flow' in reports_display.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=reports_display.index,
                        y=reports_display['Free Cash Flow'],
                        mode='lines+markers+text',
                        text=[f"${val:,.0f}" for val in reports_display['Free Cash Flow']],
                        textposition='top center'
                    ))
                    
                    fig.update_layout(
                        title="Free Cash Flow Trend",
                        xaxis_title="Year",
                        yaxis_title="Free Cash Flow ($)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    # Add a zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # FCF margin if net income is available
                    if 'Net Income' in reports_display.columns and 'Free Cash Flow' in reports_display.columns:
                        fcf_to_ni = reports_display['Free Cash Flow'] / reports_display['Net Income']
                        
                        st.markdown("#### FCF to Net Income Ratio")
                        st.write("""
                        This ratio shows how much of a company's net income is converted to free cash flow. 
                        A ratio > 1 indicates that the company generates more free cash flow than accounting profits, which is generally positive.
                        """)
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=reports_display.index,
                            y=fcf_to_ni,
                            text=[f"{val:.2f}x" for val in fcf_to_ni],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="Free Cash Flow to Net Income Ratio",
                            xaxis_title="Year",
                            yaxis_title="Ratio",
                            height=300,
                            template="plotly_white"
                        )
                        
                        # Add a line at 1.0
                        fig.add_hline(y=1.0, line_dash="dash", line_color="green")
                        
                        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying cash flow: {str(e)}")
        st.error(f"Error displaying cash flow: {str(e)}")

def calculate_financial_ratios(income_statement, balance_sheet, stock_data, company_info):
    """
    Calculate key financial ratios from financial statements.
    
    Args:
        income_statement (pd.DataFrame): Income statement data
        balance_sheet (pd.DataFrame): Balance sheet data
        stock_data (pd.DataFrame): Stock price data
        company_info (dict): Company information
        
    Returns:
        dict: Dictionary of financial ratios
    """
    try:
        ratios = {}
        
        # Profitability Ratios
        if income_statement is not None and not income_statement.empty:
            latest_income = income_statement.iloc[0]
            
            # Gross Margin
            if 'grossProfit' in latest_income and 'totalRevenue' in latest_income:
                gross_profit = float(latest_income['grossProfit'])
                revenue = float(latest_income['totalRevenue'])
                if revenue > 0:
                    ratios['Gross Margin'] = gross_profit / revenue
            
            # Operating Margin
            if 'operatingIncome' in latest_income and 'totalRevenue' in latest_income:
                operating_income = float(latest_income['operatingIncome'])
                revenue = float(latest_income['totalRevenue'])
                if revenue > 0:
                    ratios['Operating Margin'] = operating_income / revenue
            
            # Net Profit Margin
            if 'netIncome' in latest_income and 'totalRevenue' in latest_income:
                net_income = float(latest_income['netIncome'])
                revenue = float(latest_income['totalRevenue'])
                if revenue > 0:
                    ratios['Net Profit Margin'] = net_income / revenue
        
        # Leverage Ratios
        if balance_sheet is not None and not balance_sheet.empty:
            latest_balance = balance_sheet.iloc[0]
            
            # Debt to Equity
            if 'totalLiabilities' in latest_balance and 'totalShareholderEquity' in latest_balance:
                total_liabilities = float(latest_balance['totalLiabilities'])
                shareholder_equity = float(latest_balance['totalShareholderEquity'])
                if shareholder_equity > 0:
                    ratios['Debt to Equity'] = total_liabilities / shareholder_equity
            
            # Debt to Assets
            if 'totalLiabilities' in latest_balance and 'totalAssets' in latest_balance:
                total_liabilities = float(latest_balance['totalLiabilities'])
                total_assets = float(latest_balance['totalAssets'])
                if total_assets > 0:
                    ratios['Debt to Assets'] = total_liabilities / total_assets
            
            # Current Ratio
            if 'totalCurrentAssets' in latest_balance and 'totalCurrentLiabilities' in latest_balance:
                current_assets = float(latest_balance['totalCurrentAssets'])
                current_liabilities = float(latest_balance['totalCurrentLiabilities'])
                if current_liabilities > 0:
                    ratios['Current Ratio'] = current_assets / current_liabilities
        
        # Valuation Ratios
        if company_info:
            # P/E Ratio
            if 'forwardPE' in company_info:
                ratios['Forward P/E'] = company_info.get('forwardPE')
            
            # Price to Book
            if 'priceToBook' in company_info:
                ratios['Price to Book'] = company_info.get('priceToBook')
            
            # Dividend Yield
            if 'dividendYield' in company_info:
                ratios['Dividend Yield'] = company_info.get('dividendYield')
        
        # Return ratios if any were calculated
        if ratios:
            return ratios
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {str(e)}")
        return None

def display_financial_ratios(ratios):
    """
    Display financial ratios in a formatted layout.
    
    Args:
        ratios (dict): Dictionary of financial ratios
    """
    try:
        if not ratios:
            st.warning("Financial ratio data not available.")
            return
            
        st.subheader("Financial Ratios")
        
        # Group ratios by category
        profitability_ratios = {k: v for k, v in ratios.items() if k in ['Gross Margin', 'Operating Margin', 'Net Profit Margin']}
        leverage_ratios = {k: v for k, v in ratios.items() if k in ['Debt to Equity', 'Debt to Assets', 'Current Ratio']}
        valuation_ratios = {k: v for k, v in ratios.items() if k in ['Forward P/E', 'Price to Book', 'Dividend Yield']}
        
        # Create columns for different ratio categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Profitability")
            for ratio, value in profitability_ratios.items():
                if isinstance(value, (int, float)):
                    st.metric(ratio, f"{value:.2%}")
                else:
                    st.metric(ratio, "N/A")
        
        with col2:
            st.markdown("#### Leverage")
            for ratio, value in leverage_ratios.items():
                if isinstance(value, (int, float)):
                    st.metric(ratio, f"{value:.2f}")
                else:
                    st.metric(ratio, "N/A")
        
        with col3:
            st.markdown("#### Valuation")
            for ratio, value in valuation_ratios.items():
                if isinstance(value, (int, float)):
                    if ratio == 'Dividend Yield':
                        st.metric(ratio, f"{value:.2%}")
                    else:
                        st.metric(ratio, f"{value:.2f}")
                else:
                    st.metric(ratio, "N/A")
        
        # Add descriptions for ratio interpretation
        with st.expander("Financial Ratio Interpretation"):
            st.markdown("""
            ### Profitability Ratios
            - **Gross Margin**: Percentage of revenue retained after direct costs. Higher is better.
            - **Operating Margin**: Percentage of revenue retained after operating expenses. Higher is better.
            - **Net Profit Margin**: Percentage of revenue retained after all expenses. Higher is better.
            
            ### Leverage Ratios
            - **Debt to Equity**: Total liabilities divided by shareholder equity. Lower means less financial risk.
            - **Debt to Assets**: Total liabilities divided by total assets. Lower means less financial risk.
            - **Current Ratio**: Current assets divided by current liabilities. Higher than 1 indicates good short-term liquidity.
            
            ### Valuation Ratios
            - **Forward P/E**: Price per share divided by expected earnings per share. Lower might indicate undervaluation.
            - **Price to Book**: Price per share divided by book value per share. Lower might indicate undervaluation.
            - **Dividend Yield**: Annual dividends per share divided by price per share. Higher yields provide more income.
            """)
        
    except Exception as e:
        logger.error(f"Error displaying financial ratios: {str(e)}")
        st.error(f"Error displaying financial ratios: {str(e)}")
