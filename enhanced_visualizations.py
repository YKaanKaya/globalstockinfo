import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedVisualizations:
    """Enhanced visualization class with modern interactive charts."""
    
    def __init__(self):
        self.theme_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Stock Price", 
                               show_volume: bool = True, indicators: Dict = None) -> go.Figure:
        """Create an advanced candlestick chart with technical indicators."""
        
        if data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=3 if indicators else 2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[title, 'Volume'] + (['Technical Indicators'] if indicators else []),
                row_heights=[0.6, 0.2, 0.2] if indicators else [0.7, 0.3]
            )
        else:
            fig = make_subplots(
                rows=2 if indicators else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[title] + (['Technical Indicators'] if indicators else []),
                row_heights=[0.7, 0.3] if indicators else [1.0]
            )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                line=dict(width=1),
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], 
                          line=dict(color='orange', width=2),
                          name='SMA 20'),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'], 
                          line=dict(color='blue', width=2),
                          name='SMA 50'),
                row=1, col=1
            )
        
        if 'EMA_12' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['EMA_12'], 
                          line=dict(color='purple', width=1, dash='dash'),
                          name='EMA 12'),
                row=1, col=1
            )
        
        # Add Bollinger Bands if available
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Upper'], 
                          line=dict(color='gray', width=1),
                          name='BB Upper', showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Lower'], 
                          line=dict(color='gray', width=1),
                          fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                          name='Bollinger Bands'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Middle'], 
                          line=dict(color='gray', width=1, dash='dot'),
                          name='BB Middle', showlegend=False),
                row=1, col=1
            )
        
        # Add volume chart
        if show_volume and 'Volume' in data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(x=data.index, y=data['Volume'],
                      marker_color=colors, opacity=0.7,
                      name='Volume'),
                row=2, col=1
            )
        
        # Add technical indicators
        if indicators and show_volume:
            indicator_row = 3
        elif indicators and not show_volume:
            indicator_row = 2
        else:
            indicator_row = None
        
        if indicator_row and 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'],
                          line=dict(color='purple', width=2),
                          name='RSI'),
                row=indicator_row, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought", row=indicator_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold", row=indicator_row, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_rangeslider_visible=False,
            height=800 if (show_volume and indicators) else (600 if show_volume or indicators else 500),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, l=60, r=60, b=60),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=indicator_row if indicator_row else (2 if show_volume else 1), col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if indicator_row:
            fig.update_yaxes(title_text="RSI", row=indicator_row, col=1)
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create a correlation heatmap."""
        
        if data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Calculate correlation matrix
        correlation = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_risk_return_scatter(self, portfolio_data: Dict, benchmark_data: Dict = None,
                                  title: str = "Risk vs Return Analysis") -> go.Figure:
        """Create risk-return scatter plot."""
        
        fig = go.Figure()
        
        if portfolio_data:
            fig.add_trace(go.Scatter(
                x=[portfolio_data.get('annualized_volatility', 0) * 100],
                y=[portfolio_data.get('annualized_return', 0) * 100],
                mode='markers+text',
                marker=dict(size=15, color='blue', symbol='diamond'),
                text=['Portfolio'],
                textposition='top center',
                name='Portfolio',
                hovertemplate='<b>Portfolio</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
            ))
        
        if benchmark_data:
            fig.add_trace(go.Scatter(
                x=[benchmark_data.get('volatility', 0) * 100],
                y=[benchmark_data.get('return', 0) * 100],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='circle'),
                text=['S&P 500'],
                textposition='top center',
                name='S&P 500',
                hovertemplate='<b>S&P 500</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
            ))
        
        # Add efficient frontier line (theoretical)
        if portfolio_data:
            risk_range = np.linspace(0, max(portfolio_data.get('annualized_volatility', 0.2) * 100 * 1.5, 25), 100)
            efficient_return = np.sqrt(risk_range) * 2  # Simplified efficient frontier
            
            fig.add_trace(go.Scatter(
                x=risk_range,
                y=efficient_return,
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                name='Theoretical Efficient Frontier',
                hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Return (%)",
            height=500,
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def create_sector_allocation_pie(self, sector_data: Dict, title: str = "Sector Allocation") -> go.Figure:
        """Create sector allocation pie chart."""
        
        if not sector_data:
            return go.Figure().add_annotation(text="No sector data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        sectors = list(sector_data.keys())
        values = list(sector_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=sectors,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(line=dict(color='#000000', width=1))
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_performance_comparison(self, data: Dict[str, pd.DataFrame], 
                                    title: str = "Performance Comparison") -> go.Figure:
        """Create performance comparison chart."""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, df) in enumerate(data.items()):
            if not df.empty and 'Close' in df.columns:
                # Normalize to 100 at start
                normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=name,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base 100)",
            height=500,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_drawdown_chart(self, returns: pd.Series, title: str = "Drawdown Analysis") -> go.Figure:
        """Create drawdown chart."""
        
        if returns.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Calculate cumulative returns and drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Cumulative Returns', 'Drawdown (%)'],
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                      line=dict(color='blue', width=2),
                      name='Cumulative Returns',
                      hovertemplate='Date: %{x}<br>Value: %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown,
                      fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                      line=dict(color='red', width=1),
                      name='Drawdown',
                      hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=600,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_options_analysis(self, calls: pd.DataFrame, puts: pd.DataFrame,
                               current_price: float, title: str = "Options Analysis") -> go.Figure:
        """Create options analysis visualization."""
        
        if calls.empty and puts.empty:
            return go.Figure().add_annotation(text="No options data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Calls - Volume vs Open Interest', 'Puts - Volume vs Open Interest',
                           'Calls - Implied Volatility', 'Puts - Implied Volatility'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Calls volume vs OI
        if not calls.empty:
            fig.add_trace(
                go.Scatter(x=calls['strike'], y=calls['volume'],
                          mode='markers', marker=dict(size=8, color='green'),
                          name='Calls Volume',
                          hovertemplate='Strike: %{x}<br>Volume: %{y}<extra></extra>'),
                row=1, col=1
            )
            
            if 'openInterest' in calls.columns:
                fig.add_trace(
                    go.Scatter(x=calls['strike'], y=calls['openInterest'],
                              mode='markers', marker=dict(size=6, color='lightgreen'),
                              name='Calls OI',
                              hovertemplate='Strike: %{x}<br>Open Interest: %{y}<extra></extra>'),
                    row=1, col=1
                )
        
        # Puts volume vs OI
        if not puts.empty:
            fig.add_trace(
                go.Scatter(x=puts['strike'], y=puts['volume'],
                          mode='markers', marker=dict(size=8, color='red'),
                          name='Puts Volume',
                          hovertemplate='Strike: %{x}<br>Volume: %{y}<extra></extra>'),
                row=1, col=2
            )
            
            if 'openInterest' in puts.columns:
                fig.add_trace(
                    go.Scatter(x=puts['strike'], y=puts['openInterest'],
                              mode='markers', marker=dict(size=6, color='lightcoral'),
                              name='Puts OI',
                              hovertemplate='Strike: %{x}<br>Open Interest: %{y}<extra></extra>'),
                    row=1, col=2
                )
        
        # Implied Volatility
        if not calls.empty and 'impliedVolatility' in calls.columns:
            fig.add_trace(
                go.Scatter(x=calls['strike'], y=calls['impliedVolatility'],
                          mode='lines+markers', line=dict(color='green'),
                          name='Calls IV',
                          hovertemplate='Strike: %{x}<br>IV: %{y:.2f}<extra></extra>'),
                row=2, col=1
            )
        
        if not puts.empty and 'impliedVolatility' in puts.columns:
            fig.add_trace(
                go.Scatter(x=puts['strike'], y=puts['impliedVolatility'],
                          mode='lines+markers', line=dict(color='red'),
                          name='Puts IV',
                          hovertemplate='Strike: %{x}<br>IV: %{y:.2f}<extra></extra>'),
                row=2, col=2
            )
        
        # Add current price line
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=current_price, line_dash="dash", line_color="blue",
                             annotation_text=f"Current: ${current_price:.2f}",
                             row=row, col=col)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=700,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_financial_metrics_radar(self, metrics: Dict, title: str = "Financial Metrics Radar") -> go.Figure:
        """Create radar chart for financial metrics."""
        
        if not metrics:
            return go.Figure().add_annotation(text="No metrics data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Normalize metrics to 0-100 scale for radar chart
        metric_names = []
        metric_values = []
        
        # Define key metrics and their ideal ranges
        key_metrics = {
            'ROE': (metrics.get('roe', 0), 15),  # Good ROE is around 15%
            'ROA': (metrics.get('roa', 0), 5),   # Good ROA is around 5%
            'Gross Margin': (metrics.get('gross_margin', 0), 40),  # Good gross margin varies by industry
            'Operating Margin': (metrics.get('operating_margin', 0), 15),
            'Profit Margin': (metrics.get('profit_margin', 0), 10),
            'Current Ratio': (min(metrics.get('current_ratio', 0), 5), 2.5),  # Cap at 5 for visualization
        }
        
        for name, (value, ideal) in key_metrics.items():
            metric_names.append(name)
            # Normalize: 0-ideal range maps to 0-100
            normalized_value = min((value / ideal) * 100, 100) if ideal > 0 else 0
            metric_values.append(normalized_value)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=metric_values + [metric_values[0]],  # Close the polygon
            theta=metric_names + [metric_names[0]],
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.3)',
            line=dict(color='rgb(0, 100, 200)', width=2),
            name='Company Metrics'
        ))
        
        # Add ideal performance circle
        ideal_values = [100] * len(metric_names)
        fig.add_trace(go.Scatterpolar(
            r=ideal_values + [ideal_values[0]],
            theta=metric_names + [metric_names[0]],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
            name='Ideal Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_earnings_forecast_chart(self, forecast_data: pd.DataFrame, 
                                     title: str = "Earnings Forecast") -> go.Figure:
        """Create earnings forecast visualization."""
        
        if forecast_data.empty:
            return go.Figure().add_annotation(text="No forecast data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        # Historical earnings (if available)
        if 'actual' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['actual'],
                mode='lines+markers',
                name='Historical Earnings',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
        
        # Forecasted earnings
        if 'forecast' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['forecast'],
                mode='lines+markers',
                name='Forecasted Earnings',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        # Confidence intervals (if available)
        if all(col in forecast_data.columns for col in ['upper_bound', 'lower_bound']):
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 165, 0, 0.3)',
                fill='tonexty',
                name='Forecast Range',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title="Period",
            yaxis_title="Earnings Per Share ($)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig

# Global instance
enhanced_viz = EnhancedVisualizations()