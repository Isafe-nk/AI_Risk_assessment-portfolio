import numpy as np
import pandas as pd
import streamlit as st


class TechnicalAnalyzer:
    """
    Class for handling technical analysis of stocks.
    This centralizes all technical indicator calculations and related functionality.
    """
    
    def __init__(self, api_client):
        """Initialize the technical analyzer with an API client."""
        self.api_client = api_client
    
    def get_indicators(self, ticker):
        """
        Get a comprehensive set of technical indicators for a given ticker.
        Returns a dictionary of indicators.
        """
        try:
            # RSI
            rsi_data = self.api_client.get_technical_indicator(ticker, 'rsi', 14)
            
            # Get historical prices for moving averages and other indicators
            prices = self.api_client.get_historical_prices(ticker)
            
            if prices is None or prices.empty:
                return {
                    'RSI': 50,  # Default values
                    'MA50': 0,
                    'MA200': 0,
                    'Volume_Average': 0,
                    'Volume_Current': 0,
                    'MACD': 0,
                    'Signal_Line': 0,
                    'Bollinger_Upper': 0,
                    'Bollinger_Lower': 0,
                    'ATR': 0
                }
                
            # Calculate moving averages
            ma50 = self._calculate_ma(prices, 50)
            ma200 = self._calculate_ma(prices, 200)
            
            # Calculate volume metrics
            vol_avg = self._calculate_volume_average(prices, 30)
            vol_current = prices['volume'].iloc[0] if not prices.empty else 0
            
            # Calculate MACD
            macd, signal_line = self._calculate_macd(prices)
            
            # Calculate Bollinger Bands
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(prices)
            
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(prices)
            
            return {
                'RSI': rsi_data[0]['rsi'] if rsi_data else 50,
                'MA50': ma50,
                'MA200': ma200,
                'Volume_Average': vol_avg,
                'Volume_Current': vol_current,
                'MACD': macd,
                'Signal_Line': signal_line,
                'Bollinger_Upper': bollinger_upper,
                'Bollinger_Lower': bollinger_lower,
                'ATR': atr
            }
        except Exception as e:
            st.warning(f"Error getting technical indicators for {ticker}: {e}")
            return None
    
    def calculate_technical_score(self, indicators):
        """
        Calculate a technical analysis score based on the provided indicators.
        Score range: 0 (very bearish) to 100 (very bullish)
        """
        if indicators is None:
            return 50  # Neutral score if no indicators are available
            
        score = 50  # Start with a neutral score
        
        # RSI Analysis
        rsi = indicators.get('RSI')
        if rsi is not None:
            if rsi < 30:  # Oversold (bullish signal)
                score += 20
            elif rsi < 40:
                score += 10
            elif rsi > 70:  # Overbought (bearish signal)
                score -= 20
            elif rsi > 60:
                score -= 10
        
        # Moving Average Analysis
        ma50 = indicators.get('MA50')
        ma200 = indicators.get('MA200')
        if ma50 is not None and ma200 is not None and ma50 > 0 and ma200 > 0:
            if ma50 > ma200:  # Golden Cross (bullish signal)
                score += 15
            else:  # Death Cross (bearish signal)
                score -= 15
        
        # Volume Analysis
        vol_avg = indicators.get('Volume_Average')
        vol_current = indicators.get('Volume_Current')
        if vol_avg is not None and vol_current is not None and vol_avg > 0:
            ratio = vol_current / vol_avg
            if ratio > 1.5:  # High volume (could be a signal)
                # If price is going up, high volume is bullish
                # If price is going down, high volume is bearish
                if ma50 > ma200:  # Using MA as a trend indicator
                    score += 10
                else:
                    score -= 5
        
        # MACD Analysis
        macd = indicators.get('MACD')
        signal_line = indicators.get('Signal_Line')
        if macd is not None and signal_line is not None:
            if macd > signal_line:  # Bullish signal
                score += 10
            elif macd < signal_line:  # Bearish signal
                score -= 10
        
        # Bollinger Bands Analysis
        price = indicators.get('Current_Price')
        bollinger_upper = indicators.get('Bollinger_Upper')
        bollinger_lower = indicators.get('Bollinger_Lower')
        if price and bollinger_upper and bollinger_lower:
            # Price near lower band may indicate oversold (bullish)
            if price < bollinger_lower * 1.05:
                score += 10
            # Price near upper band may indicate overbought (bearish)
            elif price > bollinger_upper * 0.95:
                score -= 10
        
        # Ensure the score is within the valid range
        return max(0, min(100, score))
    
    def calculate_volatility(self, ticker, days=30):
        """Calculate N-day annualized volatility for a stock"""
        try:
            prices_df = self.api_client.get_historical_prices(ticker)
            if prices_df is None or prices_df.empty or len(prices_df) < days:
                return None
                
            # Calculate daily returns
            prices_df['return'] = prices_df['close'].pct_change()
            
            # Drop NaN values
            returns = prices_df['return'].dropna()
            
            if len(returns) < days:
                return None
                
            # Calculate standard deviation of returns
            std_dev = returns.head(days).std()
            
            # Annualize the volatility (√252 is the annualization factor for daily returns)
            annualized_volatility = std_dev * np.sqrt(252)
            return annualized_volatility
        except Exception as e:
            st.warning(f"Error calculating volatility for {ticker}: {str(e)}")
            return None
    
    def calculate_beta(self, ticker, market_index='^GSPC'):
        """Calculate beta relative to S&P 500"""
        try:
            stock_prices = self.api_client.get_historical_prices(ticker)
            market_prices = self.api_client.get_historical_prices(market_index)
            
            if stock_prices is None or market_prices is None or stock_prices.empty or market_prices.empty:
                # If market index data is not available (e.g., due to API limitations),
                # return a default beta based on sector
                return 1.0  # Default to market beta
            
            stock_returns = stock_prices['close'].pct_change().dropna()
            market_returns = market_prices['close'].pct_change().dropna()
            
            common_dates = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns[common_dates]
            market_returns = market_returns[common_dates]
            
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance != 0 else 1.0
        except Exception as e:
            st.warning(f"Error calculating beta for {ticker}: {str(e)}")
            return 1.0  # Return market beta as fallback
    
    def detect_patterns(self, ticker):
        """Detect common chart patterns"""
        try:
            prices = self.api_client.get_historical_prices(ticker)
            if prices is None or prices.empty or len(prices) < 20:
                return {}
                
            patterns = {}
            
            # Check for Double Top pattern
            patterns['double_top'] = self._detect_double_top(prices)
            
            # Check for Double Bottom pattern
            patterns['double_bottom'] = self._detect_double_bottom(prices)
            
            # Check for Head and Shoulders pattern
            patterns['head_shoulders'] = self._detect_head_shoulders(prices)
            
            # Check for Inverse Head and Shoulders pattern
            patterns['inverse_head_shoulders'] = self._detect_inverse_head_shoulders(prices)
            
            # Check for Flag/Pennant pattern
            patterns['flag_pennant'] = self._detect_flag_pennant(prices)
            
            return patterns
        except Exception as e:
            st.warning(f"Error detecting patterns for {ticker}: {e}")
            return {}
    
    def _calculate_ma(self, prices, period):
        """Calculate moving average for specified period"""
        try:
            if prices is None or prices.empty or 'close' not in prices.columns:
                return 0
                
            if len(prices) >= period:
                # Sort prices by date in ascending order for correct MA calculation
                prices_sorted = prices.sort_values('date')
                ma = prices_sorted['close'].rolling(window=period).mean()
                # Get the most recent value (last value in the series)
                return ma.iloc[-1]
            return 0
        except Exception:
            return 0
    
    def _calculate_volume_average(self, prices, period=30):
        """Calculate average volume over specified period"""
        try:
            if len(prices) >= period:
                return prices['volume'].rolling(window=period).mean().iloc[0]
            return None
        except Exception:
            return None
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            # Calculate the fast and slow EMAs
            fast_ema = prices['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate the MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate the signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Get the most recent values
            macd = macd_line.iloc[0]
            signal = signal_line.iloc[0]
            
            return macd, signal
        except Exception:
            return 0, 0
    
    def _calculate_bollinger_bands(self, prices, period=20, num_std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            if len(prices) >= period:
                # Calculate the simple moving average
                sma = prices['close'].rolling(window=period).mean()
                
                # Calculate the standard deviation
                std_dev = prices['close'].rolling(window=period).std()
                
                # Calculate upper and lower bands
                upper_band = sma + (std_dev * num_std_dev)
                lower_band = sma - (std_dev * num_std_dev)
                
                return upper_band.iloc[0], lower_band.iloc[0]
            return None, None
        except Exception:
            return None, None
    
    def _calculate_atr(self, prices, period=14):
        """Calculate Average True Range (ATR)"""
        try:
            if len(prices) > period:
                # Calculate True Range
                prices['high_low'] = prices['high'] - prices['low']
                prices['high_close'] = abs(prices['high'] - prices['close'].shift(1))
                prices['low_close'] = abs(prices['low'] - prices['close'].shift(1))
                
                prices['tr'] = prices[['high_low', 'high_close', 'low_close']].max(axis=1)
                
                # Calculate ATR
                atr = prices['tr'].rolling(window=period).mean().iloc[0]
                return atr
            return None
        except Exception:
            return None
    
    def _detect_double_top(self, prices):
        """Detect Double Top pattern"""
        # This is a simplified implementation
        # A proper implementation would consider more factors
        try:
            # Get high prices
            highs = prices['high'].values
            
            # Find local peaks
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            # Check for double top pattern
            if len(peaks) >= 2:
                peak1 = peaks[-2]
                peak2 = peaks[-1]
                
                # Peaks should be similar height
                height_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                
                # Peaks should be separated
                separation = peak2[0] - peak1[0]
                
                if height_diff < 0.03 and separation > 10:
                    return True
            
            return False
        except Exception:
            return False
    
    def _detect_double_bottom(self, prices):
        """Detect Double Bottom pattern"""
        try:
            # Get low prices
            lows = prices['low'].values
            
            # Find local troughs
            troughs = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            # Check for double bottom pattern
            if len(troughs) >= 2:
                trough1 = troughs[-2]
                trough2 = troughs[-1]
                
                # Troughs should be similar height
                height_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                
                # Troughs should be separated
                separation = trough2[0] - trough1[0]
                
                if height_diff < 0.03 and separation > 10:
                    return True
            
            return False
        except Exception:
            return False
    
    def _detect_head_shoulders(self, prices):
        """Detect Head and Shoulders pattern"""
        # Simplified implementation
        try:
            # Get high prices
            highs = prices['high'].values
            
            # Find local peaks
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            # Check for head and shoulders pattern
            if len(peaks) >= 3:
                left_shoulder = peaks[-3]
                head = peaks[-2]
                right_shoulder = peaks[-1]
                
                # Head should be higher than shoulders
                if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                    # Shoulders should be at similar heights
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                    
                    if shoulder_diff < 0.05:
                        return True
            
            return False
        except Exception:
            return False
    
    def _detect_inverse_head_shoulders(self, prices):
        """Detect Inverse Head and Shoulders pattern"""
        try:
            # Get low prices
            lows = prices['low'].values
            
            # Find local troughs
            troughs = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            # Check for inverse head and shoulders pattern
            if len(troughs) >= 3:
                left_shoulder = troughs[-3]
                head = troughs[-2]
                right_shoulder = troughs[-1]
                
                # Head should be lower than shoulders
                if head[1] < left_shoulder[1] and head[1] < right_shoulder[1]:
                    # Shoulders should be at similar heights
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                    
                    if shoulder_diff < 0.05:
                        return True
            
            return False
        except Exception:
            return False
    
    def _detect_flag_pennant(self, prices):
        """Detect Flag/Pennant pattern"""
        try:
            # This is a simplified check for a flag pattern
            # Look for strong move followed by consolidation
            
            close_prices = prices['close'].values
            volumes = prices['volume'].values
            
            # Calculate short-term trend
            short_trend = close_prices[0] - close_prices[5]
            
            # Look for high volume at start of trend
            initial_volume = volumes[5:10].mean()
            recent_volume = volumes[0:5].mean()
            
            # Flag pattern: strong move followed by consolidation with lower volume
            if abs(short_trend) > prices['close'].std() * 2 and recent_volume < initial_volume * 0.8:
                return True
                
            return False
        except Exception:
            return False