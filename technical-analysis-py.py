import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_returns(data):
    """
    Calculate daily and cumulative returns.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        
    Returns:
        pd.DataFrame: Data with added 'Daily Return' and 'Cumulative Return' columns
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute returns")
            return data
            
        # Calculate daily returns
        data['Daily Return'] = data['Close'].pct_change()
        
        # Calculate cumulative returns
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod() - 1
        
        logger.info("Successfully computed returns")
        return data
    except Exception as e:
        logger.error(f"Error computing returns: {str(e)}")
        return data

def compute_moving_averages(data, windows=[20, 50, 200]):
    """
    Calculate simple moving averages for specified windows.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        windows (list): List of periods for moving averages
        
    Returns:
        pd.DataFrame: Data with added MA columns
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute moving averages")
            return data
            
        for window in windows:
            data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
            
        logger.info(f"Successfully computed moving averages for windows {windows}")
        return data
    except Exception as e:
        logger.error(f"Error computing moving averages: {str(e)}")
        return data

def compute_exponential_moving_averages(data, windows=[12, 26, 50, 200]):
    """
    Calculate exponential moving averages for specified windows.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        windows (list): List of periods for EMAs
        
    Returns:
        pd.DataFrame: Data with added EMA columns
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute EMAs")
            return data
            
        for window in windows:
            data[f'EMA{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
            
        logger.info(f"Successfully computed EMAs for windows {windows}")
        return data
    except Exception as e:
        logger.error(f"Error computing EMAs: {str(e)}")
        return data

def compute_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        window (int): Period for moving average
        num_std (int): Number of standard deviations for bands
        
    Returns:
        pd.DataFrame: Data with added Bollinger Bands columns
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute Bollinger Bands")
            return data
            
        # Calculate middle band (simple moving average)
        data['BB_middle'] = data['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        data['BB_std'] = data['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * num_std)
        data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * num_std)
        
        # Calculate Bandwidth
        data['BB_bandwidth'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # Calculate %B (where price is in relation to the bands)
        data['BB_percent_b'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        
        logger.info(f"Successfully computed Bollinger Bands (window={window}, std={num_std})")
        return data
    except Exception as e:
        logger.error(f"Error computing Bollinger Bands: {str(e)}")
        return data

def compute_rsi(data, window=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        window (int): RSI period
        
    Returns:
        pd.DataFrame: Data with added RSI column
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute RSI")
            return data
            
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        
        # Calculate average gain and loss over the specified window
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        data['RSI'] = 100 - (100 / (1 + rs))
        
        logger.info(f"Successfully computed RSI (window={window})")
        return data
    except Exception as e:
        logger.error(f"Error computing RSI: {str(e)}")
        return data

def compute_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal EMA period
        
    Returns:
        pd.DataFrame: Data with added MACD columns
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute MACD")
            return data
            
        # Calculate fast and slow EMAs
        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        
        # Calculate signal line
        data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram/divergence
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        logger.info(f"Successfully computed MACD (fast={fast}, slow={slow}, signal={signal})")
        return data
    except Exception as e:
        logger.error(f"Error computing MACD: {str(e)}")
        return data

def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data (pd.DataFrame): Stock price data with 'High', 'Low', 'Close' columns
        k_window (int): %K period
        d_window (int): %D period
        
    Returns:
        pd.DataFrame: Data with added Stochastic Oscillator columns
    """
    try:
        required_columns = ['High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns {missing_columns} in data, cannot compute Stochastic Oscillator")
            return data
            
        # Calculate %K
        lowest_low = data['Low'].rolling(window=k_window).min()
        highest_high = data['High'].rolling(window=k_window).max()
        data['%K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (moving average of %K)
        data['%D'] = data['%K'].rolling(window=d_window).mean()
        
        logger.info(f"Successfully computed Stochastic Oscillator (K={k_window}, D={d_window})")
        return data
    except Exception as e:
        logger.error(f"Error computing Stochastic Oscillator: {str(e)}")
        return data

def compute_average_true_range(data, window=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pd.DataFrame): Stock price data with 'High', 'Low', 'Close' columns
        window (int): ATR period
        
    Returns:
        pd.DataFrame: Data with added ATR column
    """
    try:
        required_columns = ['High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns {missing_columns} in data, cannot compute ATR")
            return data
            
        # Calculate True Range
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift(1))
        low_close_prev = abs(data['Low'] - data['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        data['ATR'] = tr.rolling(window=window).mean()
        
        logger.info(f"Successfully computed ATR (window={window})")
        return data
    except Exception as e:
        logger.error(f"Error computing ATR: {str(e)}")
        return data

def compute_on_balance_volume(data):
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' and 'Volume' columns
        
    Returns:
        pd.DataFrame: Data with added OBV column
    """
    try:
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns {missing_columns} in data, cannot compute OBV")
            return data
            
        # Calculate daily price change
        data['price_change'] = data['Close'].diff()
        
        # Initialize OBV with first volume value
        data['OBV'] = 0
        
        # Calculate OBV
        for i in range(1, len(data)):
            if data['price_change'].iloc[i] > 0:
                data['OBV'].iloc[i] = data['OBV'].iloc[i-1] + data['Volume'].iloc[i]
            elif data['price_change'].iloc[i] < 0:
                data['OBV'].iloc[i] = data['OBV'].iloc[i-1] - data['Volume'].iloc[i]
            else:
                data['OBV'].iloc[i] = data['OBV'].iloc[i-1]
        
        # Drop temporary column
        data.drop('price_change', axis=1, inplace=True)
        
        logger.info("Successfully computed OBV")
        return data
    except Exception as e:
        logger.error(f"Error computing OBV: {str(e)}")
        return data

def compute_price_channels(data, window=20):
    """
    Calculate Price Channels.
    
    Args:
        data (pd.DataFrame): Stock price data with 'High' and 'Low' columns
        window (int): Period for the channel
        
    Returns:
        pd.DataFrame: Data with added upper and lower channel columns
    """
    try:
        required_columns = ['High', 'Low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns {missing_columns} in data, cannot compute Price Channels")
            return data
            
        # Calculate upper channel (highest high)
        data['upper_channel'] = data['High'].rolling(window=window).max()
        
        # Calculate lower channel (lowest low)
        data['lower_channel'] = data['Low'].rolling(window=window).min()
        
        # Calculate middle channel
        data['middle_channel'] = (data['upper_channel'] + data['lower_channel']) / 2
        
        logger.info(f"Successfully computed Price Channels (window={window})")
        return data
    except Exception as e:
        logger.error(f"Error computing Price Channels: {str(e)}")
        return data

def compute_volatility(data, window=20):
    """
    Calculate historical volatility.
    
    Args:
        data (pd.DataFrame): Stock price data with 'Close' column
        window (int): Period for volatility calculation
        
    Returns:
        pd.DataFrame: Data with added volatility column
    """
    try:
        if 'Close' not in data.columns:
            logger.warning("Missing 'Close' column in data, cannot compute volatility")
            return data
            
        # Calculate daily returns
        if 'Daily Return' not in data.columns:
            data['Daily Return'] = data['Close'].pct_change()
            
        # Calculate rolling standard deviation of returns
        data['Volatility'] = data['Daily Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        logger.info(f"Successfully computed volatility (window={window})")
        return data
    except Exception as e:
        logger.error(f"Error computing volatility: {str(e)}")
        return data

def apply_technical_indicators(data, indicators=None):
    """
    Apply selected technical indicators to the data.
    
    Args:
        data (pd.DataFrame): Stock price data
        indicators (list): List of indicators to calculate
        
    Returns:
        pd.DataFrame: Data with added technical indicator columns
    """
    try:
        logger.info(f"Applying technical indicators: {indicators}")
        
        # Always compute these basics
        data = compute_returns(data)
        data = compute_moving_averages(data)
        
        # Apply selected indicators if provided
        if indicators is not None:
            if "RSI" in indicators:
                data = compute_rsi(data)
                
            if "MACD" in indicators:
                data = compute_macd(data)
                
            if "Bollinger Bands" in indicators:
                data = compute_bollinger_bands(data)
                
            if "Stochastic" in indicators:
                data = compute_stochastic_oscillator(data)
                
            if "ATR" in indicators:
                data = compute_average_true_range(data)
                
            if "OBV" in indicators:
                data = compute_on_balance_volume(data)
                
            if "Price Channels" in indicators:
                data = compute_price_channels(data)
                
            if "Volatility" in indicators:
                data = compute_volatility(data)
                
        logger.info("Successfully applied technical indicators")
        return data
    except Exception as e:
        logger.error(f"Error applying technical indicators: {str(e)}")
        return data

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
        # Get the most recent data point
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2] if len(data) > 1 else None
        
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
            histogram = last_row['MACD_hist']
            
            macd_signal = "Neutral"
            macd_desc = "MACD shows momentum and potential trend changes."
            
            if prev_row is not None:
                # Check for crossovers
                if last_row['MACD'] > last_row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']:
                    macd_signal = "Buy (Bullish Crossover)"
                elif last_row['MACD'] < last_row['MACD_signal'] and prev_row['MACD'] >= prev_row['MACD_signal']:
                    macd_signal = "Sell (Bearish Crossover)"
                # Check histogram direction
                elif last_row['MACD_hist'] > 0 and last_row['MACD_hist'] > prev_row['MACD_hist']:
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
            
            ma_signal = "Neutral"
            ma_desc = "Moving averages show trend direction and potential support/resistance levels."
            
            if prev_row is not None:
                # Check for golden cross (MA50 crosses above MA200)
                if last_row['MA50'] > last_row['MA200'] and prev_row['MA50'] <= prev_row['MA200']:
                    ma_signal = "Strong Buy (Golden Cross)"
                # Check for death cross (MA50 crosses below MA200)
                elif last_row['MA50'] < last_row['MA200'] and prev_row['MA50'] >= prev_row['MA200']:
                    ma_signal = "Strong Sell (Death Cross)"
                # Current position
                elif ma50 > ma200:
                    ma_signal = "Bullish Trend"
                elif ma50 < ma200:
                    ma_signal = "Bearish Trend"
            
            signals["Moving Averages"] = {
                "value": {
                    "MA50": round(ma50, 2),
                    "MA200": round(ma200, 2)
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
