import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_sentiment(text):
    """
    Analyze sentiment of a given text using TextBlob.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary with sentiment analysis results
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            category = "Positive"
        elif polarity < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
            
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "category": category
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {"polarity": 0, "subjectivity": 0, "category": "Neutral"}

def analyze_news_sentiment(news):
    """
    Analyze sentiment of a list of news articles.
    
    Args:
        news (list): List of news articles
        
    Returns:
        list: List of news articles with sentiment analysis
    """
    try:
        if not news:
            return []
            
        analyzed_news = []
        for article in news:
            # Analyze title sentiment
            title_sentiment = analyze_sentiment(article['title'])
            
            # Create a copy of the article with sentiment data
            article_with_sentiment = article.copy()
            article_with_sentiment['sentiment'] = title_sentiment
            
            analyzed_news.append(article_with_sentiment)
            
        return analyzed_news
    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {str(e)}")
        return []

def get_sentiment_score(news):
    """
    Calculate an overall sentiment score based on recent news.
    
    Args:
        news (list): List of news articles
        
    Returns:
        dict: Dictionary with overall sentiment analysis
    """
    try:
        if not news:
            logger.warning("No news found for sentiment analysis")
            return {"category": "Neutral", "score": 0, "subjectivity": 0}
        
        # Analyze sentiment for each article
        analyzed_news = analyze_news_sentiment(news)
        
        # Extract sentiment scores
        sentiment_scores = [article['sentiment']['polarity'] for article in analyzed_news]
        subjectivity_scores = [article['sentiment']['subjectivity'] for article in analyzed_news]
        
        # Calculate averages
        average_score = np.mean(sentiment_scores)
        average_subjectivity = np.mean(subjectivity_scores)
        
        # Determine category
        if average_score > 0.1:
            category = "Positive"
        elif average_score < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
        
        # Track sentiment over time if timestamps are available
        sentiment_timeline = None
        if all('providerPublishTime' in article for article in analyzed_news):
            # Create timeline data
            timeline_data = [
                {
                    'timestamp': datetime.fromtimestamp(article['providerPublishTime']),
                    'polarity': article['sentiment']['polarity'],
                    'title': article['title']
                }
                for article in analyzed_news
            ]
            
            # Sort by timestamp
            timeline_data.sort(key=lambda x: x['timestamp'])
            
            sentiment_timeline = timeline_data
        
        logger.info(f"Calculated sentiment: {category} (Score: {average_score:.2f})")
        return {
            "category": category,
            "score": average_score,
            "subjectivity": average_subjectivity,
            "timeline": sentiment_timeline
        }
    except Exception as e:
        logger.error(f"Error calculating sentiment score: {str(e)}")
        return {"category": "Neutral", "score": 0, "subjectivity": 0}

def display_news(news):
    """
    Display news articles with sentiment analysis.
    
    Args:
        news (list): List of news articles
    """
    try:
        if not news:
            st.warning("No recent news available.")
            return
            
        # Analyze sentiment for the news
        analyzed_news = analyze_news_sentiment(news)
        
        st.subheader("Latest News")
        
        # Add filter for news sentiment
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Neutral", "Negative"],
            index=0
        )
        
        # Create placeholders for positive, neutral, and negative news
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        
        displayed_articles = 0
        
        for article in analyzed_news:
            sentiment = article['sentiment']
            
            # Update sentiment counts
            if sentiment['category'] == "Positive":
                positive_count += 1
            elif sentiment['category'] == "Negative":
                negative_count += 1
            else:
                neutral_count += 1
            
            # Apply sentiment filter
            if sentiment_filter != "All" and sentiment['category'] != sentiment_filter:
                continue
                
            # Define color based on sentiment
            if sentiment['category'] == "Positive":
                sentiment_color = "rgba(0, 128, 0, 0.1)"
                emoji = "📈"
            elif sentiment['category'] == "Negative":
                sentiment_color = "rgba(255, 0, 0, 0.1)"
                emoji = "📉"
            else:
                sentiment_color = "rgba(128, 128, 128, 0.1)"
                emoji = "📊"
                
            # Create a card for each news item
            with st.container():
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {sentiment_color}; margin-bottom: 10px;">
                        <h4>{article['title']}</h4>
                        <p><i>{datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}</i></p>
                        <p>{emoji} Sentiment: {sentiment['category']} (Score: {sentiment['polarity']:.2f})</p>
                        <a href="{article['link']}" target="_blank">Read more</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            displayed_articles += 1
            
            # Limit the number of displayed articles to improve performance
            if displayed_articles >= 10:
                st.info(f"Showing 10 out of {len(analyzed_news)} articles. Filter by sentiment to see different articles.")
                break
        
        # Display sentiment distribution
        st.subheader("News Sentiment Distribution")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[positive_count, neutral_count, negative_count],
                marker=dict(colors=['green', 'gray', 'red'])
            )
        ])
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying news: {str(e)}")
        st.error(f"Error displaying news: {str(e)}")

def display_sentiment_analysis(sentiment_data):
    """
    Display detailed sentiment analysis.
    
    Args:
        sentiment_data (dict): Sentiment analysis data
    """
    try:
        if not sentiment_data:
            st.warning("No sentiment data available for analysis.")
            return
            
        st.subheader("Sentiment Analysis")
        
        # Create a gauge chart for sentiment score
        score = sentiment_data.get("score", 0)
        category = sentiment_data.get("category", "Neutral")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain=dict(x=[0, 1], y=[0, 1]),
            gauge=dict(
                axis=dict(range=[-1, 1]),
                bar=dict(color="darkblue"),
                steps=[
                    dict(range=[-1, -0.1], color="red"),
                    dict(range=[-0.1, 0.1], color="gray"),
                    dict(range=[0.1, 1], color="green")
                ],
                threshold=dict(
                    line=dict(color="black", width=4),
                    thickness=0.75,
                    value=score
                )
            ),
            title=dict(text="Sentiment Score")
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment information
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment Category", category)
            st.metric("Subjectivity", f"{sentiment_data.get('subjectivity', 0):.2f}")
        
        with col2:
            st.markdown("#### Sentiment Interpretation")
            
            if category == "Positive":
                st.success("Recent news has a positive sentiment, which may have a favorable impact on stock price.")
            elif category == "Negative":
                st.error("Recent news has a negative sentiment, which may adversely affect stock price.")
            else:
                st.info("Recent news has a neutral sentiment, suggesting market indecision or balanced viewpoints.")
                
            st.markdown(f"""
            **Score: {score:.2f}** 
            - Range: -1 (very negative) to +1 (very positive)
            - 0 represents neutral sentiment
            
            **Subjectivity: {sentiment_data.get('subjectivity', 0):.2f}**
            - Range: 0 (objective) to 1 (subjective)
            - Higher values indicate more opinion-based content
            """)
        
        # Display sentiment timeline if available
        timeline_data = sentiment_data.get("timeline")
        if timeline_data:
            st.subheader("Sentiment Timeline")
            
            # Extract data for chart
            timestamps = [entry['timestamp'] for entry in timeline_data]
            polarities = [entry['polarity'] for entry in timeline_data]
            titles = [entry['title'] for entry in timeline_data]
            
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=polarities,
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=polarities,
                    colorscale='RdYlGn',
                    cmin=-1,
                    cmax=1
                ),
                text=titles,
                hovertemplate='%{text}<br>Sentiment: %{y:.2f}<br>%{x}'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Add reference lines for positive/negative thresholds
            fig.add_hline(y=0.1, line_dash="dot", line_color="rgba(0,128,0,0.3)")
            fig.add_hline(y=-0.1, line_dash="dot", line_color="rgba(255,0,0,0.3)")
            
            fig.update_layout(
                title="Sentiment Score Over Time",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[-1, 1]),
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sentiment trend interpretation
            latest_sentiments = polarities[-min(5, len(polarities)):]
            trend = np.mean(np.diff(latest_sentiments)) if len(latest_sentiments) > 1 else 0
            
            st.markdown("#### Sentiment Trend Analysis")
            
            if trend > 0.05:
                st.success("Sentiment is trending positively in recent news articles.")
            elif trend < -0.05:
                st.error("Sentiment is trending negatively in recent news articles.")
            else:
                st.info("Sentiment is relatively stable in recent news articles.")
                
    except Exception as e:
        logger.error(f"Error displaying sentiment analysis: {str(e)}")
        st.error(f"Error displaying sentiment analysis: {str(e)}")

def analyze_sentiment_impact(stock_data, news):
    """
    Analyze the potential impact of sentiment on stock price.
    
    Args:
        stock_data (pd.DataFrame): Stock price data
        news (list): List of news articles
        
    Returns:
        dict: Sentiment impact analysis
    """
    try:
        if stock_data is None or stock_data.empty or not news:
            return None
            
        # Get sentiment scores for news
        analyzed_news = analyze_news_sentiment(news)
        
        # Calculate correlation between sentiment and price changes
        # First, align news timestamps with stock data
        sentiment_df = pd.DataFrame([
            {
                'date': datetime.fromtimestamp(article['providerPublishTime']).date(),
                'sentiment': article['sentiment']['polarity']
            }
            for article in analyzed_news
            if 'providerPublishTime' in article
        ])
        
        # Group by date and calculate average sentiment
        if not sentiment_df.empty:
            sentiment_by_date = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
            
            # Convert stock data index to date for merging
            stock_df = stock_data.copy()
            stock_df['date'] = stock_df.index.date
            stock_df['price_change_pct'] = stock_df['Close'].pct_change()
            
            # Merge sentiment with stock data
            merged_data = pd.merge(
                sentiment_by_date,
                stock_df[['date', 'price_change_pct']],
                on='date',
                how='inner'
            )
            
            # Calculate correlation if we have enough data points
            if len(merged_data) >= 3:
                correlation = merged_data['sentiment'].corr(merged_data['price_change_pct'])
                
                # Create lagged sentiment (sentiment's effect on next day's price)
                merged_data['next_day_price_change'] = merged_data['price_change_pct'].shift(-1)
                lagged_correlation = merged_data['sentiment'].corr(merged_data['next_day_price_change'])
                
                return {
                    'correlation': correlation,
                    'lagged_correlation': lagged_correlation,
                    'data': merged_data
                }
        
        return None
    except Exception as e:
        logger.error(f"Error analyzing sentiment impact: {str(e)}")
        return None

def display_sentiment_impact(impact_data):
    """
    Display sentiment impact analysis.
    
    Args:
        impact_data (dict): Sentiment impact analysis data
    """
    try:
        if impact_data is None:
            st.info("Insufficient data to analyze sentiment impact on price.")
            return
            
        st.subheader("Sentiment Impact Analysis")
        
        correlation = impact_data.get('correlation')
        lagged_correlation = impact_data.get('lagged_correlation')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Same-Day Correlation", f"{correlation:.2f}" if correlation is not None else "N/A")
            
            if correlation is not None:
                if abs(correlation) < 0.2:
                    st.markdown("Weak relationship between news sentiment and price changes.")
                elif abs(correlation) < 0.4:
                    st.markdown("Moderate relationship between news sentiment and price changes.")
                else:
                    st.markdown("Strong relationship between news sentiment and price changes.")
        
        with col2:
            st.metric("Next-Day Correlation", f"{lagged_correlation:.2f}" if lagged_correlation is not None else "N/A")
            
            if lagged_correlation is not None:
                if abs(lagged_correlation) < 0.2:
                    st.markdown("Weak predictive relationship for next-day price changes.")
                elif abs(lagged_correlation) < 0.4:
                    st.markdown("Moderate predictive relationship for next-day price changes.")
                else:
                    st.markdown("Strong predictive relationship for next-day price changes.")
        
        # Display scatter plot of sentiment vs. price changes
        data = impact_data.get('data')
        if data is not None and len(data) > 3:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['sentiment'],
                y=data['price_change_pct'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=data['price_change_pct'],
                    colorscale='RdYlGn',
                    cmin=-0.05,
                    cmax=0.05
                ),
                text=[date.strftime('%Y-%m-%d') for date in data['date']],
                hovertemplate='Date: %{text}<br>Sentiment: %{x:.2f}<br>Price Change: %{y:.2%}'
            ))
            
            # Add trendline
            if len(data) >= 5:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    data['sentiment'], 
                    data['price_change_pct']
                )
                
                x_range = np.linspace(data['sentiment'].min(), data['sentiment'].max(), 100)
                y_range = slope * x_range + intercept
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name=f'Trend (r²={r_value**2:.2f})'
                ))
            
            fig.update_layout(
                title="Sentiment vs. Price Change",
                xaxis_title="News Sentiment Score",
                yaxis_title="Price Change %",
                yaxis_tickformat='.1%',
                height=400,
                template="plotly_white"
            )
            
            # Add zero lines
            fig.add_hline(y=0, line_dash="solid", line_color="gray")
            fig.add_vline(x=0, line_dash="solid", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("#### Interpretation")
            
            st.markdown("""
            This chart shows the relationship between news sentiment and price changes:
            
            - **Points in the upper right**: Positive sentiment and positive price change
            - **Points in the lower left**: Negative sentiment and negative price change
            - **Points in the upper left**: Negative sentiment but positive price change
            - **Points in the lower right**: Positive sentiment but negative price change
            
            The trend line shows the overall relationship between sentiment and price changes.
            """)
            
            # Add predictive insight based on recent sentiment
            if lagged_correlation is not None and abs(lagged_correlation) > 0.3:
                recent_sentiment = data.sort_values('date', ascending=False)['sentiment'].iloc[0]
                
                st.subheader("Predictive Insight")
                
                if recent_sentiment > 0.1 and lagged_correlation > 0.3:
                    st.success("Recent positive sentiment suggests potential upward price movement.")
                elif recent_sentiment < -0.1 and lagged_correlation > 0.3:
                    st.error("Recent negative sentiment suggests potential downward price movement.")
                elif recent_sentiment > 0.1 and lagged_correlation < -0.3:
                    st.error("Recent positive sentiment suggests potential downward price movement (contrarian indicator).")
                elif recent_sentiment < -0.1 and lagged_correlation < -0.3:
                    st.success("Recent negative sentiment suggests potential upward price movement (contrarian indicator).")
                else:
                    st.info("Recent sentiment is neutral or lacks strong predictive signal.")
            
    except Exception as e:
        logger.error(f"Error displaying sentiment impact: {str(e)}")
        st.error(f"Error displaying sentiment impact: {str(e)}")
