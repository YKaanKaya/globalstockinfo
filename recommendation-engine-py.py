import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_recommendation(ticker, company_info, esg_data, sentiment_score, data, analyst_consensus):
    """
    Generate stock recommendation based on various factors.
    
    Args:
        ticker (str): Stock ticker symbol
        company_info (dict): Company information
        esg_data (pd.DataFrame): ESG data
        sentiment_score (dict): Sentiment analysis results
        data (pd.DataFrame): Stock price data with technical indicators
        analyst_consensus (dict): Analyst recommendations consensus
        
    Returns:
        tuple: (recommendation, factors) - Recommendation and factors considered
    """
    try:
        score = 0
        factors = {}
        
        # Define factor weights
        weights = {
            'P/E Ratio': 1,
            'Price to Book': 0.75,
            'Dividend Yield': 0.5,
            'ESG Score': 0.5,
            'Sentiment': 1,
            'Technical': 1.5,
            'Analyst Consensus': 2
        }
        
        # 1. P/E Ratio (Adjusted thresholds for industry norms)
        if company_info:
            forward_pe = company_info.get('forwardPE')
            if isinstance(forward_pe, (int, float)) and forward_pe > 0:
                # Check if P/E is lower than industry average (simplified logic)
                if forward_pe < 15:
                    factors['P/E Ratio'] = 'Positive (Below Average)'
                    score += 1 * weights['P/E Ratio']
                elif 15 <= forward_pe <= 30:
                    factors['P/E Ratio'] = 'Neutral (Average Range)'
                else:
                    factors['P/E Ratio'] = 'Negative (Above Average)'
                    score -= 1 * weights['P/E Ratio']
            
            # 2. Price to Book
            price_to_book = company_info.get('priceToBook')
            if isinstance(price_to_book, (int, float)) and price_to_book > 0:
                if price_to_book < 3:
                    factors['Price to Book'] = 'Positive (Below 3)'
                    score += 1 * weights['Price to Book']
                elif 3 <= price_to_book <= 5:
                    factors['Price to Book'] = 'Neutral (3-5 Range)'
                else:
                    factors['Price to Book'] = 'Negative (Above 5)'
                    score -= 1 * weights['Price to Book']
            
            # 3. Dividend Yield
            dividend_yield = company_info.get('dividendYield')
            if isinstance(dividend_yield, (int, float)):
                if dividend_yield > 0.03:
                    factors['Dividend Yield'] = 'Positive (>3%)'
                    score += 1 * weights['Dividend Yield']
                elif 0.01 <= dividend_yield <= 0.03:
                    factors['Dividend Yield'] = 'Neutral (1-3%)'
                elif dividend_yield > 0:
                    factors['Dividend Yield'] = 'Slight Positive (>0%)'
                    score += 0.5 * weights['Dividend Yield']
                else:
                    factors['Dividend Yield'] = 'Neutral (No Dividend)'
        
        # 4. ESG Score
        if esg_data is not None:
            if 'totalEsg' in esg_data.index:
                esg_score = esg_data.loc['totalEsg'].values[0]
                
                if esg_score > 70:
                    factors['ESG Score'] = 'Very Positive (>70)'
                    score += 1.5 * weights['ESG Score']
                elif esg_score > 50:
                    factors['ESG Score'] = 'Positive (50-70)'
                    score += 1 * weights['ESG Score']
                elif esg_score > 30:
                    factors['ESG Score'] = 'Neutral (30-50)'
                else:
                    factors['ESG Score'] = 'Negative (<30)'
                    score -= 1 * weights['ESG Score']
        
        # 5. Sentiment Score
        if sentiment_score:
            sentiment_category = sentiment_score.get('category', 'Neutral')
            sentiment_value = sentiment_score.get('score', 0)
            
            if sentiment_category == 'Positive':
                if sentiment_value > 0.3:
                    factors['Sentiment'] = 'Very Positive (>0.3)'
                    score += 1.5 * weights['Sentiment']
                else:
                    factors['Sentiment'] = 'Positive (0.1-0.3)'
                    score += 1 * weights['Sentiment']
            elif sentiment_category == 'Negative':
                if sentiment_value < -0.3:
                    factors['Sentiment'] = 'Very Negative (<-0.3)'
                    score -= 1.5 * weights['Sentiment']
                else:
                    factors['Sentiment'] = 'Negative (-0.1 to -0.3)'
                    score -= 1 * weights['Sentiment']
            else:
                factors['Sentiment'] = 'Neutral'
        
        # 6. Technical Indicators
        if data is not None and not data.empty:
            technical_score = 0
            tech_factors = []
            
            # 6.1 RSI
            if 'RSI' in data.columns:
                latest_rsi = data['RSI'].iloc[-1]
                if latest_rsi < 30:
                    technical_score += 1
                    tech_factors.append("RSI Oversold (<30)")
                elif latest_rsi > 70:
                    technical_score -= 1
                    tech_factors.append("RSI Overbought (>70)")
            
            # 6.2 MACD
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                latest_macd = data['MACD'].iloc[-1]
                latest_signal = data['MACD_signal'].iloc[-1]
                prev_macd = data['MACD'].iloc[-2] if len(data) > 1 else None
                prev_signal = data['MACD_signal'].iloc[-2] if len(data) > 1 else None
                
                # Bullish crossover (MACD crosses above signal line)
                if prev_macd is not None and prev_signal is not None:
                    if latest_macd > latest_signal and prev_macd <= prev_signal:
                        technical_score += 1
                        tech_factors.append("MACD Bullish Crossover")
                    # Bearish crossover (MACD crosses below signal line)
                    elif latest_macd < latest_signal and prev_macd >= prev_signal:
                        technical_score -= 1
                        tech_factors.append("MACD Bearish Crossover")
            
            # 6.3 Moving Averages
            if 'MA50' in data.columns and 'MA200' in data.columns:
                latest_ma50 = data['MA50'].iloc[-1]
                latest_ma200 = data['MA200'].iloc[-1]
                latest_close = data['Close'].iloc[-1]
                
                # Price above both MAs
                if latest_close > latest_ma50 and latest_close > latest_ma200:
                    technical_score += 0.5
                    tech_factors.append("Price Above 50 & 200 MA")
                # Price below both MAs
                elif latest_close < latest_ma50 and latest_close < latest_ma200:
                    technical_score -= 0.5
                    tech_factors.append("Price Below 50 & 200 MA")
                
                # Golden Cross (50MA crosses above 200MA)
                prev_ma50 = data['MA50'].iloc[-2] if len(data) > 1 else None
                prev_ma200 = data['MA200'].iloc[-2] if len(data) > 1 else None
                
                if prev_ma50 is not None and prev_ma200 is not None:
                    if latest_ma50 > latest_ma200 and prev_ma50 <= prev_ma200:
                        technical_score += 1.5
                        tech_factors.append("Golden Cross (50MA > 200MA)")
                    # Death Cross (50MA crosses below 200MA)
                    elif latest_ma50 < latest_ma200 and prev_ma50 >= prev_ma200:
                        technical_score -= 1.5
                        tech_factors.append("Death Cross (50MA < 200MA)")
            
            # 6.4 Bollinger Bands
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                latest_close = data['Close'].iloc[-1]
                latest_upper = data['BB_upper'].iloc[-1]
                latest_lower = data['BB_lower'].iloc[-1]
                
                if latest_close > latest_upper:
                    technical_score -= 0.5
                    tech_factors.append("Price Above Upper Bollinger")
                elif latest_close < latest_lower:
                    technical_score += 0.5
                    tech_factors.append("Price Below Lower Bollinger")
            
            # Normalize technical score to range [-1, 1]
            normalized_tech_score = max(min(technical_score, 3), -3) / 3
            
            # Add to total score
            if normalized_tech_score > 0:
                factors['Technical'] = f'Positive ({", ".join(tech_factors)})'
                score += normalized_tech_score * weights['Technical']
            elif normalized_tech_score < 0:
                factors['Technical'] = f'Negative ({", ".join(tech_factors)})'
                score += normalized_tech_score * weights['Technical']
            else:
                factors['Technical'] = f'Neutral ({", ".join(tech_factors) if tech_factors else "No Strong Signals"})'
        
        # 7. Analyst Consensus
        if analyst_consensus:
            buy_count = analyst_consensus.get('Buy', 0)
            hold_count = analyst_consensus.get('Hold', 0)
            sell_count = analyst_consensus.get('Sell', 0)
            total_count = buy_count + hold_count + sell_count
            
            if total_count > 0:
                buy_ratio = buy_count / total_count
                sell_ratio = sell_count / total_count
                
                if buy_ratio > 0.7:
                    factors['Analyst Consensus'] = f'Very Positive ({buy_count}/{total_count} Buy)'
                    score += 1.5 * weights['Analyst Consensus']
                elif buy_ratio > 0.5:
                    factors['Analyst Consensus'] = f'Positive ({buy_count}/{total_count} Buy)'
                    score += 1 * weights['Analyst Consensus']
                elif sell_ratio > 0.7:
                    factors['Analyst Consensus'] = f'Very Negative ({sell_count}/{total_count} Sell)'
                    score -= 1.5 * weights['Analyst Consensus']
                elif sell_ratio > 0.5:
                    factors['Analyst Consensus'] = f'Negative ({sell_count}/{total_count} Sell)'
                    score -= 1 * weights['Analyst Consensus']
                else:
                    factors['Analyst Consensus'] = f'Neutral ({hold_count}/{total_count} Hold)'
        
        # Generate recommendation based on total score
        max_possible_score = sum([v for k, v in weights.items() if k in factors])
        min_possible_score = -max_possible_score
        score_range = max_possible_score - min_possible_score
        
        # Normalize score to percentage of maximum possible
        if score_range > 0:
            normalized_score = (score - min_possible_score) / score_range
        else:
            normalized_score = 0.5  # Default to neutral if no range
        
        # Determine recommendation
        if normalized_score > 0.80:
            recommendation = "Strong Buy"
        elif normalized_score > 0.60:
            recommendation = "Buy"
        elif normalized_score > 0.45:
            recommendation = "Mild Buy"
        elif normalized_score < 0.20:
            recommendation = "Strong Sell"
        elif normalized_score < 0.40:
            recommendation = "Sell"
        elif normalized_score < 0.55:
            recommendation = "Hold"
        else:
            recommendation = "Hold"
        
        # Add normalized score to the factors
        factors['_normalized_score'] = normalized_score
        factors['_raw_score'] = score
        
        logger.info(f"Generated recommendation for {ticker}: {recommendation}")
        return recommendation, factors
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        return "Hold", {"Error": f"Could not generate recommendation: {str(e)}"}

def display_recommendation_visualization(recommendation, factors):
    """
    Display visualization of the recommendation and factors.
    
    Args:
        recommendation (str): The stock recommendation
        factors (dict): Dictionary of factors considered for the recommendation
    """
    try:
        st.subheader("Stock Recommendation Analysis")
        
        # Display the recommendation prominently
        recommendation_color = "green" if "Buy" in recommendation else "red" if "Sell" in recommendation else "blue"
        st.markdown(f"<h2 style='color: {recommendation_color};'>{recommendation}</h2>", unsafe_allow_html=True)
        
        # Extract normalized score
        normalized_score = factors.pop('_normalized_score', 0.5) if '_normalized_score' in factors else 0.5
        raw_score = factors.pop('_raw_score', 0) if '_raw_score' in factors else 0
        
        # Create a gauge chart for the overall recommendation
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=normalized_score * 100,
            domain=dict(x=[0, 1], y=[0, 1]),
            gauge=dict(
                axis=dict(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100]),
                bar=dict(color="darkblue"),
                bgcolor="white",
                steps=[
                    dict(range=[0, 20], color="darkred"),
                    dict(range=[20, 40], color="coral"),
                    dict(range=[40, 60], color="gold"),
                    dict(range=[60, 80], color="yellowgreen"),
                    dict(range=[80, 100], color="green")
                ],
                threshold=dict(
                    line=dict(color="black", width=2),
                    thickness=0.75,
                    value=normalized_score * 100
                )
            ),
            title=dict(text="Recommendation Score")
        ))
        
        # Add annotations for recommendation zones
        fig.add_annotation(x=0.1, y=0.2, text="Strong Sell", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.3, y=0.2, text="Sell", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.5, y=0.2, text="Hold", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.7, y=0.2, text="Buy", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.9, y=0.2, text="Strong Buy", showarrow=False, font=dict(size=10))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=70, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display factor breakdown
        st.subheader("Factor Analysis")
        
        # Convert factors to numerical scores for visualization
        factor_scores = []
        factor_names = []
        factor_details = []
        
        for factor, assessment in factors.items():
            if 'Positive' in assessment:
                if 'Very Positive' in assessment:
                    score = 2
                else:
                    score = 1
            elif 'Negative' in assessment:
                if 'Very Negative' in assessment:
                    score = -2
                else:
                    score = -1
            else:
                score = 0
            
            factor_names.append(factor)
            factor_scores.append(score)
            factor_details.append(assessment)
        
        # Create color map for scores
        colors = {
            2: 'rgba(0, 128, 0, 0.8)',    # Dark green for very positive
            1: 'rgba(144, 238, 144, 0.8)', # Light green for positive
            0: 'rgba(169, 169, 169, 0.8)', # Gray for neutral
            -1: 'rgba(255, 99, 71, 0.8)',  # Light red for negative
            -2: 'rgba(178, 34, 34, 0.8)'   # Dark red for very negative
        }
        
        bar_colors = [colors[score] for score in factor_scores]
        
        # Create a horizontal bar chart for factor scores
        factor_fig = go.Figure(go.Bar(
            x=factor_scores,
            y=factor_names,
            orientation='h',
            marker_color=bar_colors,
            text=factor_details,
            textposition='outside',
            hoverinfo='text',
            hovertext=factor_details
        ))
        
        factor_fig.update_layout(
            title="Factor Contribution to Recommendation",
            xaxis=dict(
                title="Impact",
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                range=[-2.5, 2.5]
            ),
            yaxis=dict(
                title=None,
                categoryorder='total ascending'
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white"
        )
        
        # Add a vertical line at x=0
        factor_fig.add_vline(x=0, line_width=1, line_color="black", line_dash="dash")
        
        st.plotly_chart(factor_fig, use_container_width=True)
        
        # Display a detailed table of factors
        st.markdown("### Factor Details")
        
        factor_df = pd.DataFrame({
            "Factor": factor_names,
            "Assessment": factor_details,
            "Impact": ["Very Positive" if score == 2 else 
                      "Positive" if score == 1 else 
                      "Neutral" if score == 0 else 
                      "Negative" if score == -1 else 
                      "Very Negative" for score in factor_scores]
        })
        
        # Apply color formatting to the table
        def color_impact(val):
            color_map = {
                "Very Positive": "background-color: rgba(0, 128, 0, 0.2)",
                "Positive": "background-color: rgba(144, 238, 144, 0.2)",
                "Neutral": "background-color: rgba(169, 169, 169, 0.2)",
                "Negative": "background-color: rgba(255, 99, 71, 0.2)",
                "Very Negative": "background-color: rgba(178, 34, 34, 0.2)"
            }
            return color_map.get(val, "")
        
        st.dataframe(factor_df.style.applymap(color_impact, subset=['Impact']))
        
        # Display investment considerations
        st.markdown("### Investment Considerations")
        
        st.markdown("""
        **Disclaimer:** This recommendation is generated automatically based on quantitative and qualitative factors. 
        It should be considered as one of many inputs for your investment decision. Always do your own research and 
        consider consulting with a financial advisor before making investment decisions.
        """)
        
        # Add recommendation-specific considerations
        if "Buy" in recommendation:
            st.markdown("""
            #### Potential Reasons to Buy:
            - Favorable valuation metrics compared to peers or historical average
            - Positive technical indicators suggesting upward momentum
            - Analyst consensus supporting purchase
            - Positive news sentiment in recent coverage
            
            **Consider:** Position sizing, entry timing, and your overall portfolio diversification
            """)
        elif "Sell" in recommendation:
            st.markdown("""
            #### Potential Reasons to Sell:
            - Unfavorable valuation metrics suggesting overvaluation
            - Technical indicators suggesting downward momentum
            - Negative analyst outlook
            - Negative news sentiment in recent coverage
            
            **Consider:** Tax implications, alternative investments, and whether a partial position reduction might be appropriate
            """)
        else:
            st.markdown("""
            #### Reasons for Hold Recommendation:
            - Mixed signals from various factors
            - Relatively fair valuation compared to peers or historical average
            - No strong technical trends in either direction
            - Balanced analyst opinions
            
            **Consider:** Monitoring for changes in fundamentals or technical breakouts
            """)
        
    except Exception as e:
        logger.error(f"Error displaying recommendation visualization: {str(e)}")
        st.error(f"Error displaying recommendation: {str(e)}")

def generate_trading_signals(data, indicators):
    """
    Generate trading signals based on technical indicators.
    
    Args:
        data (pd.DataFrame): Stock price data with technical indicators
        indicators (list): List of indicators used
        
    Returns:
        dict: Dictionary with trading signals
    """
    signals = {}
    
    try:
        # Check if data is valid
        if data is None or data.empty or len(data) < 2:
            return {"Error": {"signal": "Error", "description": "Insufficient data for generating signals"}}
        
        # Get the most recent data points
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # RSI signals
        if "RSI" in indicators and 'RSI' in last_row:
            rsi_value = last_row['RSI']
            rsi_signal = "Neutral"
            rsi_desc = "RSI measures momentum on a scale of 0 to 100."
            
            if rsi_value < 30:
                rsi_signal = "Potential Buy (Oversold)"
            elif rsi_value > 70:
                rsi_signal = "Potential Sell (Overbought)"
                
            signals["RSI"] = {
                "value": round(rsi_value, 2),
                "signal": rsi_signal,
                "description": rsi_desc
            }
        
        # MACD signals
        if "MACD" in indicators and 'MACD' in last_row and 'MACD_signal' in last_row:
            macd = last_row['MACD']
            signal = last_row['MACD_signal']
            histogram = last_row['MACD_hist'] if 'MACD_hist' in last_row else macd - signal
            
            macd_signal = "Neutral"
            macd_desc = "MACD shows momentum and potential trend changes."
            
            # Check for crossovers
            if last_row['MACD'] > last_row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']:
                macd_signal = "Buy (Bullish Crossover)"
            elif last_row['MACD'] < last_row['MACD_signal'] and prev_row['MACD'] >= prev_row['MACD_signal']:
                macd_signal = "Sell (Bearish Crossover)"
            # Check histogram direction
            elif 'MACD_hist' in last_row and 'MACD_hist' in prev_row:
                if last_row['MACD_hist'] > 0 and last_row['MACD_hist'] > prev_row['MACD_hist']:
                    macd_signal = "Bullish (Strengthening)"
                elif last_row['MACD_hist'] < 0 and last_row['MACD_hist'] < prev_row['MACD_hist']:
                    macd_signal = "Bearish (Strengthening)"
            
            signals["MACD"] = {
                "value": {
                    "macd": round(macd, 4),
                    "signal": round(signal, 4),
                    "histogram": round(histogram, 4)
                },
                "signal": macd_signal,
                "description": macd_desc
            }
        
        # Bollinger Bands signals
        if "Bollinger Bands" in indicators and 'BB_upper' in last_row and 'BB_lower' in last_row:
            close = last_row['Close']
            upper = last_row['BB_upper']
            lower = last_row['BB_lower']
            
            bb_signal = "Neutral"
            bb_desc = "Bollinger Bands measure volatility and potential reversal points."
            
            if close > upper:
                bb_signal = "Potential Sell (Price above upper band)"
            elif close < lower:
                bb_signal = "Potential Buy (Price below lower band)"
                
            # Check for Bollinger Band squeeze (low volatility)
            if 'BB_bandwidth' in last_row:
                bandwidth = last_row['BB_bandwidth']
                if bandwidth < 0.10:  # Arbitrary threshold for low volatility
                    bb_signal += " - Volatility squeeze (potential breakout)"
            
            signals["Bollinger Bands"] = {
                "value": {
                    "upper": round(upper, 2),
                    "middle": round(last_row['BB_middle'], 2) if 'BB_middle' in last_row else "N/A",
                    "lower": round(lower, 2)
                },
                "signal": bb_signal,
                "description": bb_desc
            }
        
        # Moving Average signals
        if 'MA50' in last_row and 'MA200' in last_row:
            ma50 = last_row['MA50']
            ma200 = last_row['MA200']
            close = last_row['Close']
            
            ma_signal = "Neutral"
            ma_desc = "Moving averages show trend direction and potential support/resistance levels."
            
            # Current position relative to MAs
            if close > ma50 and close > ma200:
                ma_signal = "Bullish (Price Above Both MAs)"
            elif close < ma50 and close < ma200:
                ma_signal = "Bearish (Price Below Both MAs)"
            elif close > ma50 and close < ma200:
                ma_signal = "Mixed (Above 50MA, Below 200MA)"
            elif close < ma50 and close > ma200:
                ma_signal = "Mixed (Below 50MA, Above 200MA)"
            
            # Check for golden/death cross
            if 'MA50' in prev_row and 'MA200' in prev_row:
                if last_row['MA50'] > last_row['MA200'] and prev_row['MA50'] <= prev_row['MA200']:
                    ma_signal = "Strong Buy (Golden Cross)"
                elif last_row['MA50'] < last_row['MA200'] and prev_row['MA50'] >= prev_row['MA200']:
                    ma_signal = "Strong Sell (Death Cross)"
            
            signals["Moving Averages"] = {
                "value": {
                    "MA50": round(ma50, 2),
                    "MA200": round(ma200, 2),
                    "Close": round(close, 2)
                },
                "signal": ma_signal,
                "description": ma_desc
            }
        
        # Overall signal (combination of indicators)
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        for indicator, signal_data in signals.items():
            signal = signal_data["signal"].lower()
            if "buy" in signal or "bullish" in signal:
                bullish_signals += 1
            elif "sell" in signal or "bearish" in signal:
                bearish_signals += 1
            total_signals += 1
        
        if total_signals > 0:
            bullish_percent = (bullish_signals / total_signals) * 100
            bearish_percent = (bearish_signals / total_signals) * 100
            
            if bullish_percent >= 60:
                overall_signal = "Bullish"
            elif bearish_percent >= 60:
                overall_signal = "Bearish"
            else:
                overall_signal = "Neutral"
                
            signals["Overall"] = {
                "signal": overall_signal,
                "description": f"Combined signal based on {total_signals} indicators ({bullish_percent:.0f}% bullish, {bearish_percent:.0f}% bearish)"
            }
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        return {"Error": {"signal": "Error", "description": f"Failed to generate signals: {str(e)}"}}
