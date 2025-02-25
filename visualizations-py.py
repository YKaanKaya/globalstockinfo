import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_stock_chart(data, ticker, indicators):
    """
    Display an interactive stock chart with selected technical indicators.
    
    Args:
        data (pd.DataFrame): Stock price data with technical indicators
        ticker (str): Stock ticker symbol
        indicators (list): List of indicators to display
    """
    try:
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1, 
            subplot_titles=(f'{ticker} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'], 
                high=data['High'],
                low=data['Low'], 
                close=data['Close'],
                name='Price',
                increasing_line=dict(color='#26a69a'),
                decreasing_line=dict(color='#ef5350')
            ), 
            row=1, 
            col=1
        )

        # Add moving averages
        if 'MA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['MA20'], 
                    name='20 Day MA', 
                    line=dict(color='rgba(13, 71, 161, 0.7)', width=1.5)
                ), 
                row=1, 
                col=1
            )
        
        if 'MA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['MA50'], 
                    name='50 Day MA', 
                    line=dict(color='rgba(230, 81, 0, 0.7)', width=1.5)
                ), 
                row=1, 
                col=1
            )
        
        if 'MA200' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['MA200'], 
                    name='200 Day MA', 
                    line=dict(color='rgba(156, 39, 176, 0.7)', width=1.5)
                ), 
                row=1, 
                col=1
            )

        # Add Bollinger Bands if selected
        if "Bollinger Bands" in indicators:
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['BB_upper'], 
                        name='Upper BB', 
                        line=dict(color='rgba(0, 128, 0, 0.3)', width=1.5)
                    ), 
                    row=1, 
                    col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['BB_lower'], 
                        name='Lower BB', 
                        line=dict(color='rgba(0, 128, 0, 0.3)', width=1.5),
                        fill='tonexty', 
                        fillcolor='rgba(0, 128, 0, 0.1)'
                    ), 
                    row=1, 
                    col=1
                )

        # Add volume bars with color based on price movement
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for i, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index, 
                y=data['Volume'], 
                name='Volume',
                marker=dict(
                    color=colors,
                    line=dict(color=colors, width=1)
                )
            ), 
            row=2, 
            col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )

        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying stock chart: {str(e)}")
        st.error(f"Error displaying stock chart: {str(e)}")

def display_returns_chart(data, ticker, window=30):
    """
    Display chart of cumulative returns.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Cumulative Return' column
        ticker (str): Stock ticker symbol
        window (int): Rolling window for returns calculation
    """
    try:
        if 'Cumulative Return' not in data.columns:
            logger.warning("Missing 'Cumulative Return' column in data")
            st.warning("Cumulative return data not available for display.")
            return
            
        # Create figure
        fig = go.Figure()
        
        # Add cumulative return trace
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['Cumulative Return'], 
                mode='lines', 
                name='Cumulative Return',
                line=dict(color='rgba(0, 128, 255, 0.8)', width=2)
            )
        )
        
        # Add rolling return if data is sufficient
        if len(data) > window:
            rolling_return = data['Daily Return'].rolling(window=window).mean() * window
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=rolling_return, 
                    mode='lines', 
                    name=f'{window}-Day Rolling Return',
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5, dash='dash')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Returns Analysis',
            xaxis_title='Date',
            yaxis_title='Return',
            yaxis_tickformat='.1%',
            hovermode="x unified",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying returns chart: {str(e)}")
        st.error(f"Error displaying returns chart: {str(e)}")

def display_rsi_chart(data):
    """
    Display RSI chart with overbought/oversold zones.
    
    Args:
        data (pd.DataFrame): Stock price data with 'RSI' column
    """
    try:
        if 'RSI' not in data.columns:
            logger.warning("Missing 'RSI' column in data")
            st.warning("RSI data not available for display.")
            return
            
        # Create figure
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add overbought and oversold reference lines
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=70,
            x1=data.index[-1],
            y1=70,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=30,
            x1=data.index[-1],
            y1=30,
            line=dict(
                color="green",
                width=2,
                dash="dash",
            )
        )
        
        # Color the background of overbought/oversold regions
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[70] * len(data),
                mode='lines',
                fill=None,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[100] * len(data),
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color="rgba(0,0,0,0)"),
                name='Overbought'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[30] * len(data),
                mode='lines',
                fill=None,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=[0] * len(data),
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                line=dict(color="rgba(0,0,0,0)"),
                name='Oversold'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(
                range=[0, 100]
            ),
            height=400,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        # Add annotations
        fig.add_annotation(
            x=data.index[-1],
            y=70,
            text="Overbought (70)",
            showarrow=False,
            xshift=50,
            font=dict(color="red")
        )
        
        fig.add_annotation(
            x=data.index[-1],
            y=30,
            text="Oversold (30)",
            showarrow=False,
            xshift=50,
            font=dict(color="green")
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show interpretation
        with st.expander("RSI Interpretation"):
            st.write("""
            **Relative Strength Index (RSI)** is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.
            
            - **RSI > 70**: Generally considered overbought (potential sell signal)
            - **RSI < 30**: Generally considered oversold (potential buy signal)
            - **RSI = 50**: Neutral momentum
            
            RSI can also show:
            - **Divergence**: When price makes a new high/low but RSI doesn't, suggesting potential reversal
            - **Failure Swings**: When RSI crosses back above 30 (bullish) or below 70 (bearish)
            
            RSI works best in ranging markets and should be used with other indicators for confirmation.
            """)
        
    except Exception as e:
        logger.error(f"Error displaying RSI chart: {str(e)}")
        st.error(f"Error displaying RSI chart: {str(e)}")

def display_macd_chart(data):
    """
    Display MACD chart with signal line and histogram.
    
    Args:
        data (pd.DataFrame): Stock price data with MACD columns
    """
    try:
        if 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
            logger.warning("Missing MACD columns in data")
            st.warning("MACD data not available for display.")
            return
            
        # Create subplot with 2 rows
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1, 
            subplot_titles=("MACD and Signal Line", "MACD Histogram"),
            row_heights=[0.7, 0.3]
        )
        
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add Signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add MACD Histogram with color based on value
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_hist']]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_hist'],
                name='Histogram',
                marker=dict(
                    color=colors,
                    line=dict(color=colors, width=1)
                )
            ),
            row=2, col=1
        )
        
        # Add zero line on both plots
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=0,
            x1=data.index[-1],
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="solid",
            ),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=0,
            x1=data.index[-1],
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="solid",
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            xaxis_title="Date",
            hovermode="x unified",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Histogram", row=2, col=1)
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show interpretation
        with st.expander("MACD Interpretation"):
            st.write("""
            **Moving Average Convergence Divergence (MACD)** is a trend-following momentum indicator that shows the relationship between two moving averages.
            
            The MACD consists of:
            - **MACD Line**: The difference between the 12-day and 26-day exponential moving averages (EMAs)
            - **Signal Line**: 9-day EMA of the MACD Line
            - **Histogram**: The difference between the MACD Line and the Signal Line
            
            Common MACD signals:
            - **Crossovers**: When the MACD line crosses above the signal line (bullish) or below (bearish)
            - **Divergence**: When the price makes a new high/low but the MACD doesn't, suggesting potential reversal
            - **Rapid rises/falls**: When the MACD rises/falls rapidly, indicating overbought/oversold conditions
            
            The histogram helps visualize the crossovers and the strength of the momentum.
            """)
        
    except Exception as e:
        logger.error(f"Error displaying MACD chart: {str(e)}")
        st.error(f"Error displaying MACD chart: {str(e)}")

def display_bollinger_bands_chart(data):
    """
    Display dedicated Bollinger Bands chart.
    
    Args:
        data (pd.DataFrame): Stock price data with Bollinger Bands columns
    """
    try:
        if 'BB_upper' not in data.columns or 'BB_lower' not in data.columns:
            logger.warning("Missing Bollinger Bands columns in data")
            st.warning("Bollinger Bands data not available for display.")
            return
            
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add middle band (typically 20-day SMA)
        if 'BB_middle' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_middle'],
                    mode='lines',
                    name='Middle Band (SMA)',
                    line=dict(color='blue', width=1.5)
                )
            )
        
        # Add upper and lower bands with fill
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                mode='lines',
                name='Upper Band',
                line=dict(color='green', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                mode='lines',
                name='Lower Band',
                line=dict(color='green', width=1.5),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.1)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show interpretation
        with st.expander("Bollinger Bands Interpretation"):
            st.write("""
            **Bollinger Bands** are volatility bands placed above and below a moving average. They consist of:
            
            - **Middle Band**: A simple moving average (typically 20-day)
            - **Upper Band**: Middle band + (2 × standard deviation)
            - **Lower Band**: Middle band - (2 × standard deviation)
            
            Common signals:
            - **Bollinger Bounce**: Price tends to return to the middle band
            - **Bollinger Squeeze**: When bands narrow, indicating low volatility and potential breakout
            - **W-Bottoms**: Double bottom where the second bottom is higher than the first but still below the lower band
            - **M-Tops**: Double top where the second top is lower than the first but still above the upper band
            
            When price touches or moves outside the bands, it doesn't necessarily indicate a buy or sell signal on its own - confirmation from other indicators is recommended.
            """)
        
    except Exception as e:
        logger.error(f"Error displaying Bollinger Bands chart: {str(e)}")
        st.error(f"Error displaying Bollinger Bands chart: {str(e)}")

def create_comparison_chart(comparison_data, chart_title="Performance Comparison"):
    """
    Create an interactive chart comparing stock performances.
    
    Args:
        comparison_data (pd.DataFrame): DataFrame with returns for multiple stocks
        chart_title (str): Title for the chart
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        if comparison_data is None or comparison_data.empty:
            logger.warning("No data available for comparison chart.")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each stock with custom hover info
        for column in comparison_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_data.index, 
                    y=comparison_data[column], 
                    mode='lines', 
                    name=column,
                    hovertemplate='%{x}<br>%{y:.2f}x<extra>' + column + '</extra>'
                )
            )
        
        # Update layout with better formatting
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (1.0 = No Change)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        # Add range selector and buttons
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Add baseline at y=1.0
        fig.add_hline(
            y=1.0, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text="Baseline"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating comparison chart: {str(e)}")
        return None

def display_analyst_recommendations(consensus):
    """
    Display analyst recommendations in an interactive chart.
    
    Args:
        consensus (dict): Dictionary with analyst consensus data
    """
    try:
        if consensus is None:
            st.warning("No analyst consensus available.")
            return
            
        labels = list(consensus.keys())
        values = list(consensus.values())
        
        # Create a donut chart
        colors = {'Buy': 'green', 'Hold': 'gold', 'Sell': 'red'}
        chart_colors = [colors.get(label, 'gray') for label in labels]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels, 
                values=values, 
                hole=.4,
                marker_colors=chart_colors,
                textinfo='label+percent',
                insidetextorientation='radial',
                hoverinfo='label+value',
                texttemplate='%{label}: %{value}'
            )
        ])
        
        fig.update_layout(
            title="Analyst Recommendations Distribution",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white",
            showlegend=False
        )
        
        # Add text in the center of the donut
        total = sum(values)
        rec_text = "Analysts"
        if total > 0:
            fig.add_annotation(
                text=f"{total}<br>{rec_text}",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying analyst recommendations: {str(e)}")
        st.error(f"Error displaying analyst recommendations: {str(e)}")

def display_esg_data(esg_data):
    """
    Display ESG data in an interactive chart.
    
    Args:
        esg_data (pd.DataFrame): ESG data for a company
    """
    try:
        st.subheader("Environmental, Social, and Governance (ESG) Analysis")
        
        # Select relevant metrics
        relevant_metrics = ['totalEsg', 'environmentScore', 'socialScore', 'governanceScore']
        
        # Filter data for relevant metrics
        filtered_data = esg_data[esg_data.index.isin(relevant_metrics)]
        
        if filtered_data.empty:
            st.warning("No relevant ESG metrics found in the data.")
            return
        
        # Mapping for better metric names
        metric_names = {
            'totalEsg': 'Total ESG Score',
            'environmentScore': 'Environmental Score',
            'socialScore': 'Social Score',
            'governanceScore': 'Governance Score'
        }
        
        # Colors for each metric
        colors = {
            'Total ESG Score': 'purple',
            'Environmental Score': 'green',
            'Social Score': 'blue',
            'Governance Score': 'orange'
        }
        
        # Create a horizontal bar chart
        fig = go.Figure()
        
        # Add bars for each metric
        for metric in filtered_data.index:
            readable_metric = metric_names.get(metric, metric)
            score = filtered_data.loc[metric].values[0]
            
            fig.add_trace(
                go.Bar(
                    x=[score],
                    y=[readable_metric],
                    orientation='h',
                    name=readable_metric,
                    marker_color=colors.get(readable_metric, 'gray'),
                    text=[f"{score:.1f}"],
                    textposition='auto',
                    hoverinfo='text',
                    hovertext=[f"{readable_metric}: {score:.1f}"]
                )
            )
        
        # Update layout
        fig.update_layout(
            title="ESG Scores",
            xaxis_title="Score (0-100)",
            yaxis=dict(
                title=None,
                categoryorder='total ascending'
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white",
            showlegend=False
        )
        
        # Add reference lines for score categories
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5,
            x1=0, y1=len(filtered_data) - 0.5,
            line=dict(color="black", width=1)
        )
        
        fig.add_shape(
            type="line",
            x0=25, y0=-0.5,
            x1=25, y1=len(filtered_data) - 0.5,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=50, y0=-0.5,
            x1=50, y1=len(filtered_data) - 0.5,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=75, y0=-0.5,
            x1=75, y1=len(filtered_data) - 0.5,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Add annotations for score categories
        fig.add_annotation(
            x=12.5, y=len(filtered_data),
            text="Poor",
            showarrow=False,
            font=dict(color="red")
        )
        
        fig.add_annotation(
            x=37.5, y=len(filtered_data),
            text="Average",
            showarrow=False,
            font=dict(color="orange")
        )
        
        fig.add_annotation(
            x=62.5, y=len(filtered_data),
            text="Good",
            showarrow=False,
            font=dict(color="blue")
        )
        
        fig.add_annotation(
            x=87.5, y=len(filtered_data),
            text="Excellent",
            showarrow=False,
            font=dict(color="green")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display explanation of ESG scores
        with st.expander("Understanding ESG Scores"):
            st.write("""
            **Environmental, Social, and Governance (ESG) scores** measure a company's exposure to long-term environmental, social, and governance risks. These risks include factors that are typically not captured by traditional financial analysis.
            
            - **Environmental factors** include carbon emissions, water usage, waste management, and natural resource conservation
            - **Social factors** include employee relations, diversity, human rights, consumer protection, and community impact
            - **Governance factors** include corporate structure, executive compensation, audit committee structure, and shareholder rights
            
            **Score Interpretation:**
            - **0-25**: Poor performance, significant ESG risks
            - **26-50**: Below average performance
            - **51-75**: Above average performance
            - **76-100**: Excellent performance, well-managed ESG risks
            
            Higher scores generally indicate better management of ESG risks, which some studies suggest may correlate with long-term financial performance.
            """)
        
        # Display additional ESG metrics if available
        other_metrics = [idx for idx in esg_data.index if idx not in relevant_metrics]
        if other_metrics:
            with st.expander("Additional ESG Metrics"):
                other_data = esg_data.loc[other_metrics]
                st.dataframe(other_data.style.format("{:.2f}"))
        
    except Exception as e:
        logger.error(f"Error displaying ESG data: {str(e)}")
        st.error(f"Error displaying ESG data: {str(e)}")
