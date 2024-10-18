# data_processing.py

import pandas as pd
import numpy as np
from textblob import TextBlob
import streamlit as st

def compute_returns(data):
    """Compute daily and cumulative returns."""
    data['Daily Return'] = data['Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    return data

def compute_moving_averages(data, windows=[50, 200]):
    """Compute moving averages."""
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

def get_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    data['RSI'] = 100 - (100/(1 + rs))
    return data

def get_sentiment_score(news):
    """Calculate sentiment score based on news headlines."""
    try:
        if not news:
            return "Neutral"
        sentiment_scores = []
        for article in news[:10]:
            blob = TextBlob(article['title'])
            sentiment_scores.append(blob.sentiment.polarity)
        average_score = np.mean(sentiment_scores)
        if average_score > 0.1:
            return "Positive"
        elif average_score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error calculating sentiment: {str(e)}")
        return "Neutral"

def compute_analyst_consensus(estimates):
    """Compute analyst consensus from estimates DataFrame."""
    if estimates is None or estimates.empty:
        return None

    if 'recommendationKey' in estimates.columns:
        recs = estimates['recommendationKey'].dropna()
        # Count the number of 'buy', 'hold', 'sell' recommendations
        buy_terms = ['buy', 'strong_buy']
        hold_terms = ['hold']
        sell_terms = ['sell', 'strong_sell']

        buy_count = recs[recs.isin(buy_terms)].count()
        hold_count = recs[recs.isin(hold_terms)].count()
        sell_count = recs[recs.isin(sell_terms)].count()

        total = buy_count + hold_count + sell_count
        if total == 0:
            return None

        consensus = {
            'Buy': buy_count,
            'Hold': hold_count,
            'Sell': sell_count
        }
        return consensus
    else:
        # Process yfinance recommendations
        return process_recommendations(estimates)

def process_recommendations(recommendations):
    """Process the recommendations DataFrame to compute consensus."""
    try:
        # Normalize the 'To Grade' and 'Action' columns
        grades = []
        if 'To Grade' in recommendations.columns:
            grades.extend(recommendations['To Grade'].dropna().str.lower())
        if 'Action' in recommendations.columns:
            grades.extend(recommendations['Action'].dropna().str.lower())
        
        if not grades:
            st.warning("No recognizable recommendation data found.")
            return None

        # Map grades to categories
        buy_terms = {'buy', 'strong buy', 'outperform', 'overweight', 'add'}
        hold_terms = {'hold', 'neutral', 'equal-weight', 'maintains'}
        sell_terms = {'sell', 'strong sell', 'underperform', 'underweight', 'reduce'}

        buy_count = sum(grade in buy_terms for grade in grades)
        hold_count = sum(grade in hold_terms for grade in grades)
        sell_count = sum(grade in sell_terms for grade in grades)

        total = buy_count + hold_count + sell_count
        if total == 0:
            return None

        consensus = {
            'Buy': buy_count,
            'Hold': hold_count,
            'Sell': sell_count
        }
        return consensus
    except Exception as e:
        st.error(f"Error processing analyst recommendations: {str(e)}")
        return None

def generate_recommendation(ticker, company_info, esg_data, sentiment_score, data, analyst_consensus):
    """Generate stock recommendation based on various factors."""
    score = 0
    factors = {}
    weights = {
        'P/E Ratio': 1,
        'Dividend Yield': 0.5,
        'ESG Score': 0.5,
        'Sentiment': 1,
        'RSI': 1,
        'Analyst Consensus': 2
    }

    # P/E Ratio (Adjusted thresholds for industry norms)
    forward_pe = company_info.get('forwardPE', None)
    if isinstance(forward_pe, (int, float)) and forward_pe != 0:
        if forward_pe < 20:
            factors['P/E Ratio'] = 'Positive'
            score += 1 * weights['P/E Ratio']
        elif 20 <= forward_pe <= 40:
            factors['P/E Ratio'] = 'Neutral'
        else:
            factors['P/E Ratio'] = 'Negative'
            score -= 1 * weights['P/E Ratio']
    else:
        factors['P/E Ratio'] = 'Neutral'

    # Dividend Yield
    dividend_yield = company_info.get('dividendYield', None)
    if isinstance(dividend_yield, (int, float)):
        if dividend_yield > 0.02:
            factors['Dividend Yield'] = 'Positive'
            score += 1 * weights['Dividend Yield']
        elif 0.005 <= dividend_yield <= 0.02:
            factors['Dividend Yield'] = 'Neutral'
        else:
            factors['Dividend Yield'] = 'Negative'
            score -= 1 * weights['Dividend Yield']
    else:
        factors['Dividend Yield'] = 'Neutral'

    # ESG Score
    if esg_data is not None:
        esg_score = esg_data.loc['totalEsg'].values[0]
        if esg_score > 50:
            factors['ESG Score'] = 'Positive'
            score += 1 * weights['ESG Score']
        elif 30 <= esg_score <= 50:
            factors['ESG Score'] = 'Neutral'
        else:
            factors['ESG Score'] = 'Negative'
            score -= 1 * weights['ESG Score']
    else:
        factors['ESG Score'] = 'Neutral'

    # Sentiment Score
    if sentiment_score == 'Positive':
        factors['Sentiment'] = 'Positive'
        score += 1 * weights['Sentiment']
    elif sentiment_score == 'Neutral':
        factors['Sentiment'] = 'Neutral'
    elif sentiment_score == 'Negative':
        factors['Sentiment'] = 'Negative'
        score -= 1 * weights['Sentiment']
    else:
        factors['Sentiment'] = 'Neutral'

    # RSI Indicator
    if 'RSI' in data.columns:
        latest_rsi = data['RSI'].iloc[-1]
        if latest_rsi < 30:
            factors['RSI'] = 'Positive (Oversold)'
            score += 1 * weights['RSI']
        elif 30 <= latest_rsi <= 70:
            factors['RSI'] = 'Neutral'
        else:
            factors['RSI'] = 'Negative (Overbought)'
            score -= 1 * weights['RSI']
    else:
        factors['RSI'] = 'Neutral'

    # Analyst Consensus
    if analyst_consensus is not None:
        buy = analyst_consensus.get('Buy', 0)
        hold = analyst_consensus.get('Hold', 0)
        sell = analyst_consensus.get('Sell', 0)
        total = buy + hold + sell
        if total > 0:
            buy_ratio = buy / total
            sell_ratio = sell / total
            if buy_ratio > 0.6:
                factors['Analyst Consensus'] = 'Positive'
                score += 1 * weights['Analyst Consensus']
            elif sell_ratio > 0.6:
                factors['Analyst Consensus'] = 'Negative'
                score -= 1 * weights['Analyst Consensus']
            else:
                factors['Analyst Consensus'] = 'Neutral'
        else:
            factors['Analyst Consensus'] = 'Neutral'
    else:
        factors['Analyst Consensus'] = 'Neutral'

    # Generate Recommendation
    max_score = sum(weights.values())
    if score >= 0.6 * max_score:
        recommendation = 'Strong Buy'
    elif score >= 0.3 * max_score:
        recommendation = 'Buy'
    elif score <= -0.6 * max_score:
        recommendation = 'Strong Sell'
    elif score <= -0.3 * max_score:
        recommendation = 'Sell'
    else:
        recommendation = 'Hold'

    return recommendation, factors
