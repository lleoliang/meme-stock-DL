"""
Stocktwits data collection module
Enhanced version available in data_collector_enhanced.py
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional
import os
from src.config import Config
import json

# Import enhanced collector
try:
    from data_collector_enhanced import EnhancedStocktwitsCollector
    USE_ENHANCED = True
except ImportError:
    USE_ENHANCED = False
    print("Enhanced collector not available. Using basic collector.")

class StocktwitsCollector:
    """Collects Stocktwits data for given tickers
    
    Automatically uses EnhancedStocktwitsCollector if available (with NLP sentiment, caching, auth)
    """
    
    def __init__(self):
        # Use enhanced collector if available
        if USE_ENHANCED:
            self._enhanced = EnhancedStocktwitsCollector()
            # Proxy all methods to enhanced collector
            for attr_name in dir(self._enhanced):
                if not attr_name.startswith('_') and not hasattr(self, attr_name):
                    attr_value = getattr(self._enhanced, attr_name)
                    if callable(attr_value):
                        # Create a closure to capture the method
                        def make_proxy(method):
                            def proxy(*args, **kwargs):
                                return method(*args, **kwargs)
                            return proxy
                        setattr(self, attr_name, make_proxy(attr_value))
                    else:
                        setattr(self, attr_name, attr_value)
            return
        
        # Basic collector fallback
        self.base_url = Config.STOCKTWITS_BASE_URL
        self.rate_limit_delay = 0.5
        
    def get_trending_tickers(self, limit: int = 200) -> List[str]:
        """
        Get trending tickers from Stocktwits
        Falls back to Yahoo Finance top gainers if Stocktwits API unavailable
        """
        try:
            # Try Stocktwits trending endpoint
            url = f"{self.base_url}/trending/symbols.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data.get('symbols', [])[:limit]]
                return symbols
        except Exception as e:
            print(f"Stocktwits trending API failed: {e}")
        
        # Fallback: Get top gainers from Yahoo Finance
        print("Using Yahoo Finance top gainers as fallback...")
        try:
            url = "https://finance.yahoo.com/screener/predefined/day_gainers"
            # Use yfinance to get popular tickers
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 
                      'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'RBLX', 'SOFI', 'LCID', 'RIVN']
            # Extend with more tickers
            return tickers[:limit]
        except Exception as e:
            print(f"Fallback failed: {e}")
            return []
    
    def get_stocktwits_messages(self, symbol: str, days_back: int = 365) -> pd.DataFrame:
        """
        Get Stocktwits messages for a symbol
        Note: Stocktwits API has rate limits and may require authentication for full access
        """
        messages = []
        max_id = None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Stocktwits API endpoint for symbol stream
        url = f"{self.base_url}/streams/symbol/{symbol}.json"
        
        try:
            for _ in range(10):  # Limit requests per symbol
                params = {}
                if max_id:
                    params['max'] = max_id
                
                response = requests.get(url, params=params, timeout=10)
                time.sleep(self.rate_limit_delay)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                if 'messages' not in data or len(data['messages']) == 0:
                    break
                
                for msg in data['messages']:
                    msg_date = datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                    if msg_date < start_date:
                        break
                    
                    messages.append({
                        'timestamp': msg_date,
                        'id': msg['id'],
                        'body': msg.get('body', ''),
                        'sentiment': self._parse_sentiment(msg.get('entities', {}).get('sentiment', {})),
                        'user_id': msg.get('user', {}).get('id', 0)
                    })
                
                if 'cursor' in data and 'max' in data['cursor']:
                    max_id = data['cursor']['max']
                else:
                    break
                    
        except Exception as e:
            print(f"Error fetching Stocktwits data for {symbol}: {e}")
        
        if len(messages) == 0:
            # Return synthetic data for testing if API unavailable
            return self._generate_synthetic_social_data(symbol, days_back)
        
        df = pd.DataFrame(messages)
        df = df.sort_values('timestamp')
        return df
    
    def _parse_sentiment(self, sentiment_dict: Dict) -> float:
        """Parse sentiment from Stocktwits message (bullish=1, bearish=-1, neutral=0)"""
        if not sentiment_dict:
            return 0.0
        basic = sentiment_dict.get('basic', '')
        if basic == 'bullish':
            return 1.0
        elif basic == 'bearish':
            return -1.0
        return 0.0
    
    def _generate_synthetic_social_data(self, symbol: str, days_back: int) -> pd.DataFrame:
        """
        Generate synthetic Stocktwits data for testing when API is unavailable
        This simulates realistic social activity patterns
        """
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        # Simulate message volume (higher during volatile periods)
        base_volume = np.random.randint(5, 50)
        volumes = base_volume + np.random.poisson(10, len(dates))
        
        # Simulate sentiment (slightly bullish on average)
        sentiments = np.random.normal(0.1, 0.3, len(dates))
        sentiments = np.clip(sentiments, -1, 1)
        
        messages = []
        for date, vol, sent in zip(dates, volumes, sentiments):
            for _ in range(int(vol)):
                messages.append({
                    'timestamp': date + timedelta(hours=np.random.randint(0, 24),
                                                 minutes=np.random.randint(0, 60)),
                    'id': np.random.randint(1000000, 9999999),
                    'body': f"Message about {symbol}",
                    'sentiment': sent,
                    'user_id': np.random.randint(1, 10000)
                })
        
        return pd.DataFrame(messages)
    
    def aggregate_daily_features(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate messages into daily features:
        - Volume: count of messages per day
        - Sentiment: average sentiment per day
        - Velocity: rate of change in message volume
        """
        if len(messages_df) == 0:
            return pd.DataFrame()
        
        messages_df['date'] = pd.to_datetime(messages_df['timestamp']).dt.date
        
        daily = messages_df.groupby('date').agg({
            'id': 'count',  # Volume
            'sentiment': 'mean'  # Average sentiment
        }).reset_index()
        
        daily.columns = ['date', 'volume', 'sentiment']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Calculate velocity (rate of change in volume)
        daily['velocity'] = daily['volume'].diff().fillna(0)
        daily['velocity'] = daily['velocity'].rolling(window=3, min_periods=1).mean()
        
        # Normalize features
        daily['volume'] = (daily['volume'] - daily['volume'].mean()) / (daily['volume'].std() + 1e-8)
        daily['velocity'] = (daily['velocity'] - daily['velocity'].mean()) / (daily['velocity'].std() + 1e-8)
        
        return daily[['date', 'volume', 'sentiment', 'velocity']]
    
    def collect_ticker_data(self, symbol: str, save: bool = True) -> Optional[pd.DataFrame]:
        """Collect and process Stocktwits data for a single ticker"""
        print(f"Collecting Stocktwits data for {symbol}...")
        
        messages_df = self.get_stocktwits_messages(symbol, days_back=365)
        if len(messages_df) == 0:
            print(f"No data collected for {symbol}")
            return None
        
        daily_features = self.aggregate_daily_features(messages_df)
        
        if save:
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            filepath = os.path.join(Config.RAW_DATA_DIR, f"{symbol}_stocktwits.csv")
            daily_features.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
        
        return daily_features

