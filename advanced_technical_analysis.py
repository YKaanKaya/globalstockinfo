import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTechnicalAnalysis:
    """Advanced technical analysis with comprehensive indicators."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: float = None) -> pd.Series:
        """Exponential Moving Average."""
        if alpha is None:
            alpha = 2.0 / (window + 1.0)
        return data.ewm(alpha=alpha).mean()
    
    @staticmethod
    def wma(data: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = AdvancedTechnicalAnalysis.ema(data, fast)
        ema_slow = AdvancedTechnicalAnalysis.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = AdvancedTechnicalAnalysis.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = AdvancedTechnicalAnalysis.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def roc(data: pd.Series, window: int = 12) -> pd.Series:
        """Rate of Change."""
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    @staticmethod
    def momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """Momentum."""
        return data - data.shift(window)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index."""
        # Calculate True Range
        tr = AdvancedTechnicalAnalysis.atr(high, low, close, 1)
        
        # Calculate Directional Movement
        dm_pos = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_neg = np.where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), low.diff().abs(), 0)
        
        dm_pos = pd.Series(dm_pos, index=high.index)
        dm_neg = pd.Series(dm_neg, index=high.index)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=window).mean()
        dm_pos_smooth = dm_pos.rolling(window=window).mean()
        dm_neg_smooth = dm_neg.rolling(window=window).mean()
        
        # Calculate DI+ and DI-
        di_pos = 100 * (dm_pos_smooth / tr_smooth)
        di_neg = 100 * (dm_neg_smooth / tr_smooth)
        
        # Calculate ADX
        dx = 100 * np.abs(di_pos - di_neg) / (di_pos + di_neg)
        adx = dx.rolling(window=window).mean()
        
        return adx, di_pos, di_neg
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud."""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close plotted 26 periods in the past
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def fibonacci_retracements(high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        diff = high_price - low_price
        return {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100%': low_price
        }
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR."""
        length = len(close)
        psar = np.zeros(length)
        psarbull = np.zeros(length)
        psarbear = np.zeros(length)
        
        bull = True
        af = af_start
        hp = high.iloc[0]
        lp = low.iloc[0]
        
        for i in range(1, length):
            if bull:
                psar[i] = psar[i-1] + af * (hp - psar[i-1])
                if low.iloc[i] <= psar[i]:
                    bull = False
                    psar[i] = hp
                    lp = low.iloc[i]
                    af = af_start
                else:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + af_increment, af_max)
                psarbull[i] = psar[i] if bull else np.nan
            else:
                psar[i] = psar[i-1] + af * (lp - psar[i-1])
                if high.iloc[i] >= psar[i]:
                    bull = True
                    psar[i] = lp
                    hp = high.iloc[i]
                    af = af_start
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + af_increment, af_max)
                psarbear[i] = psar[i] if not bull else np.nan
        
        return pd.Series(psar, index=close.index)
    
    @staticmethod
    def volume_indicators(price: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        # On-Balance Volume
        obv = (volume * np.sign(price.diff())).cumsum()
        
        # Volume-Weighted Average Price (VWAP)
        typical_price = price
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Accumulation/Distribution Line
        money_flow_multiplier = ((price - price.rolling(1).min()) - (price.rolling(1).max() - price)) / (price.rolling(1).max() - price.rolling(1).min())
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        
        # Chaikin Money Flow
        cmf = money_flow_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return {
            'obv': obv,
            'vwap': vwap,
            'ad_line': ad_line,
            'cmf': cmf
        }
    
    @staticmethod
    def support_resistance_levels(data: pd.Series, window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
        """Identify support and resistance levels."""
        highs = data.rolling(window=window, center=True).max()
        lows = data.rolling(window=window, center=True).min()
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data.iloc[i] == highs.iloc[i]:
                # Check if this level has been touched multiple times
                level = data.iloc[i]
                touches = sum(1 for j in range(len(data)) if abs(data.iloc[j] - level) < level * 0.01)
                if touches >= min_touches:
                    resistance_levels.append(level)
            
            if data.iloc[i] == lows.iloc[i]:
                level = data.iloc[i]
                touches = sum(1 for j in range(len(data)) if abs(data.iloc[j] - level) < level * 0.01)
                if touches >= min_touches:
                    support_levels.append(level)
        
        return {
            'support': list(set(support_levels)),
            'resistance': list(set(resistance_levels))
        }
    
    @staticmethod
    def detect_patterns(data: pd.DataFrame) -> Dict[str, List]:
        """Detect common chart patterns."""
        patterns = {
            'double_top': [],
            'double_bottom': [],
            'head_shoulders': [],
            'triangles': [],
            'flags': [],
            'pennants': []
        }
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Simple pattern detection (can be enhanced)
        # Double Top/Bottom detection
        peaks = []
        troughs = []
        
        for i in range(1, len(close) - 1):
            if close.iloc[i] > close.iloc[i-1] and close.iloc[i] > close.iloc[i+1]:
                peaks.append((i, close.iloc[i]))
            if close.iloc[i] < close.iloc[i-1] and close.iloc[i] < close.iloc[i+1]:
                troughs.append((i, close.iloc[i]))
        
        # Double top detection
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            if abs(peak1[1] - peak2[1]) < peak1[1] * 0.03:  # Within 3%
                patterns['double_top'].append({
                    'start': peak1[0],
                    'peak1': peak1,
                    'peak2': peak2,
                    'end': peak2[0]
                })
        
        # Double bottom detection
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            if abs(trough1[1] - trough2[1]) < trough1[1] * 0.03:  # Within 3%
                patterns['double_bottom'].append({
                    'start': trough1[0],
                    'trough1': trough1,
                    'trough2': trough2,
                    'end': trough2[0]
                })
        
        return patterns
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and add them to the dataframe."""
        if data.empty:
            return data
        
        df = data.copy()
        
        try:
            # Price-based indicators
            df['SMA_20'] = AdvancedTechnicalAnalysis.sma(df['Close'], 20)
            df['SMA_50'] = AdvancedTechnicalAnalysis.sma(df['Close'], 50)
            df['SMA_200'] = AdvancedTechnicalAnalysis.sma(df['Close'], 200)
            df['EMA_12'] = AdvancedTechnicalAnalysis.ema(df['Close'], 12)
            df['EMA_26'] = AdvancedTechnicalAnalysis.ema(df['Close'], 26)
            df['WMA_20'] = AdvancedTechnicalAnalysis.wma(df['Close'], 20)
            
            # Oscillators
            df['RSI'] = AdvancedTechnicalAnalysis.rsi(df['Close'])
            df['Williams_R'] = AdvancedTechnicalAnalysis.williams_r(df['High'], df['Low'], df['Close'])
            df['CCI'] = AdvancedTechnicalAnalysis.cci(df['High'], df['Low'], df['Close'])
            df['ROC'] = AdvancedTechnicalAnalysis.roc(df['Close'])
            df['Momentum'] = AdvancedTechnicalAnalysis.momentum(df['Close'])
            
            # MACD
            macd, signal, histogram = AdvancedTechnicalAnalysis.macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = AdvancedTechnicalAnalysis.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            df['BB_Width'] = bb_upper - bb_lower
            df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch_k, stoch_d = AdvancedTechnicalAnalysis.stochastic_oscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # ATR
            df['ATR'] = AdvancedTechnicalAnalysis.atr(df['High'], df['Low'], df['Close'])
            
            # ADX
            adx, di_pos, di_neg = AdvancedTechnicalAnalysis.adx(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx
            df['DI_Plus'] = di_pos
            df['DI_Minus'] = di_neg
            
            # Ichimoku
            ichimoku = AdvancedTechnicalAnalysis.ichimoku_cloud(df['High'], df['Low'], df['Close'])
            for key, value in ichimoku.items():
                df[f'Ichimoku_{key}'] = value
            
            # Parabolic SAR
            df['PSAR'] = AdvancedTechnicalAnalysis.parabolic_sar(df['High'], df['Low'], df['Close'])
            
            # Volume indicators (if volume data is available)
            if 'Volume' in df.columns:
                volume_indicators = AdvancedTechnicalAnalysis.volume_indicators(df['Close'], df['Volume'])
                for key, value in volume_indicators.items():
                    df[f'Volume_{key}'] = value
            
            # Returns
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Volatility
            df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            logger.info("Successfully calculated all technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    @staticmethod
    def get_trading_signals(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate trading signals based on technical indicators."""
        signals = {}
        
        if data.empty:
            return signals
        
        try:
            # RSI signals
            signals['RSI_Oversold'] = data['RSI'] < 30
            signals['RSI_Overbought'] = data['RSI'] > 70
            
            # MACD signals
            signals['MACD_Bullish'] = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
            signals['MACD_Bearish'] = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
            
            # Moving Average signals
            signals['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
            signals['Death_Cross'] = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
            
            # Bollinger Band signals
            signals['BB_Squeeze'] = data['BB_Width'] < data['BB_Width'].rolling(window=20).mean()
            signals['BB_Upper_Break'] = data['Close'] > data['BB_Upper']
            signals['BB_Lower_Break'] = data['Close'] < data['BB_Lower']
            
            # Stochastic signals
            signals['Stoch_Oversold'] = (data['Stoch_K'] < 20) & (data['Stoch_D'] < 20)
            signals['Stoch_Overbought'] = (data['Stoch_K'] > 80) & (data['Stoch_D'] > 80)
            
            # ADX trend strength
            signals['Strong_Trend'] = data['ADX'] > 25
            signals['Weak_Trend'] = data['ADX'] < 20
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
        
        return signals

# Global instance
technical_analyzer = AdvancedTechnicalAnalysis()