"""
Enhanced Stocktwits data collection with:
1. API authentication support
2. NLP-based sentiment analysis
3. Caching
4. Better rate limiting
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import pickle
from pathlib import Path
import json

from config import Config
from sentiment_analyzer import SentimentAnalyzer

# Try to import scraper as fallback
try:
    from stocktwits_scraper import StocktwitsScraper
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

class EnhancedStocktwitsCollector:
    """Enhanced Stocktwits collector with authentication, NLP sentiment, and caching"""
    
    def __init__(self):
        self.api_source = Config.STOCKTWITS_API_SOURCE
        self.base_url = Config.STOCKTWITS_BASE_URL
        self.access_token = Config.STOCKTWITS_ACCESS_TOKEN
        self.rapidapi_key = Config.RAPIDAPI_KEY
        self.rapidapi_url = Config.RAPIDAPI_STOCKTWITS_URL
        self.rate_limit_delay = 0.5
        self.use_nlp = Config.USE_NLP_SENTIMENT
        self.cache_enabled = Config.CACHE_SOCIAL_DATA
        
        # Warn about API source
        if self.api_source == 'stocktwits' and not self.access_token:
            print("Note: Stocktwits has paused new API registrations.")
            print("Consider using RapidAPI (set STOCKTWITS_API_SOURCE=rapidapi) or PyTwits wrapper.")
        
        # Initialize NLP sentiment analyzer if enabled
        self.nlp_analyzer = None
        if self.use_nlp:
            try:
                self.nlp_analyzer = SentimentAnalyzer(method=Config.NLP_SENTIMENT_METHOD)
            except Exception as e:
                print(f"Warning: Could not initialize NLP sentiment analyzer: {e}")
                print("Falling back to Stocktwits built-in sentiment only")
                self.use_nlp = False
        
        # Cache directory
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication based on API source"""
        headers = {}
        
        if self.api_source == 'rapidapi':
            if not self.rapidapi_key:
                raise ValueError("RAPIDAPI_KEY not set. Get one from https://rapidapi.com/")
            headers['X-RapidAPI-Key'] = self.rapidapi_key
            headers['X-RapidAPI-Host'] = 'stocktwits-api.p.rapidapi.com'
        elif self.api_source == 'stocktwits':
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
        
        return headers
    
    def _get_api_url(self, endpoint: str) -> str:
        """Get the correct API URL based on source"""
        if self.api_source == 'rapidapi':
            # RapidAPI Stocktwits endpoints use different format
            # Common format: /api/2/streams/symbol/{symbol}
            # Try both with and without .json
            if endpoint.endswith('.json'):
                endpoint_clean = endpoint.replace('.json', '')
            else:
                endpoint_clean = endpoint
            # RapidAPI might need full path
            return f"{self.rapidapi_url}/{endpoint_clean}"
        else:
            return f"{self.base_url}/{endpoint}"
    
    def _get_cached_messages(self, symbol: str, days_back: int) -> Optional[pd.DataFrame]:
        """Check cache for messages"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{symbol}_messages_{days_back}.pkl"
        
        if cache_file.exists():
            try:
                cached_data = pickle.load(open(cache_file, 'rb'))
                cache_time = cached_data.get('timestamp', datetime.min)
                expiry_time = datetime.now() - timedelta(hours=Config.CACHE_EXPIRY_HOURS)
                
                if cache_time > expiry_time:
                    print(f"  Using cached data for {symbol} (cached {cache_time})")
                    return cached_data.get('messages')
            except Exception as e:
                print(f"  Error reading cache: {e}")
        
        return None
    
    def _save_to_cache(self, symbol: str, days_back: int, messages: pd.DataFrame):
        """Save messages to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{symbol}_messages_{days_back}.pkl"
        try:
            pickle.dump({
                'timestamp': datetime.now(),
                'messages': messages
            }, open(cache_file, 'wb'))
        except Exception as e:
            print(f"  Warning: Could not save to cache: {e}")
    
    def get_trending_tickers(self, limit: int = 200) -> List[str]:
        """Get trending tickers from Stocktwits"""
        try:
            url = self._get_api_url("trending/symbols.json")
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data.get('symbols', [])[:limit]]
                return symbols
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting...")
                time.sleep(60)
            else:
                print(f"API returned status {response.status_code}")
        except Exception as e:
            print(f"Stocktwits trending API failed: {e}")
        
        # Fallback
        print("Using fallback ticker list...")
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 
                  'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'RBLX', 'SOFI', 'LCID', 'RIVN']
        return tickers[:limit]
    
    def get_stocktwits_messages(self, symbol: str, days_back: int = 365) -> pd.DataFrame:
        """
        Get Stocktwits messages with enhanced features
        """
        # Check cache first
        cached = self._get_cached_messages(symbol, days_back)
        if cached is not None:
            return cached
        
        messages = []
        max_id = None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = self._get_api_url(f"streams/symbol/{symbol}.json")
        headers = self._get_headers()
        
        request_count = 0
        # Adjust max requests based on API source
        if self.api_source == 'rapidapi' and self.rapidapi_key:
            max_requests = 30  # RapidAPI typically allows more
        elif self.access_token:
            max_requests = 30  # More requests with Stocktwits auth
        else:
            max_requests = 10  # Public API limit
        
        try:
            while request_count < max_requests:
                params = {}
                if max_id:
                    params['max'] = max_id
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                request_count += 1
                
                # Handle rate limiting
                if response.status_code == 429:
                    print(f"  Rate limit hit for {symbol}. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                if response.status_code != 200:
                    if response.status_code == 401:
                        print(f"  Authentication failed (401). Response: {response.text[:200]}")
                        if self.api_source == 'rapidapi':
                            print(f"  Check RapidAPI key and subscription status")
                    elif response.status_code == 404:
                        print(f"  Endpoint not found (404). RapidAPI Stocktwits API may not be available.")
                        print(f"  Falling back to synthetic data for testing.")
                        # Break and use synthetic fallback
                        messages = []
                        break
                    else:
                        print(f"  API error {response.status_code}: {response.text[:200]}")
                    break
                
                data = response.json()
                if 'messages' not in data or len(data['messages']) == 0:
                    break
                
                for msg in data['messages']:
                    try:
                        msg_date = datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        if msg_date < start_date:
                            break
                        
                        body = msg.get('body', '')
                        
                        # Get sentiment
                        sentiment = self._parse_sentiment(
                            msg.get('entities', {}).get('sentiment', {}),
                            body
                        )
                        
                        messages.append({
                            'timestamp': msg_date,
                            'id': msg['id'],
                            'body': body,
                            'sentiment': sentiment,
                            'user_id': msg.get('user', {}).get('id', 0)
                        })
                    except Exception as e:
                        continue
                
                # Check for pagination
                if 'cursor' in data and 'max' in data['cursor']:
                    max_id = data['cursor']['max']
                else:
                    break
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
        except Exception as e:
            print(f"Error fetching Stocktwits data for {symbol}: {e}")
        
        if len(messages) == 0:
            # Try Selenium scraper first (most reliable)
            try:
                from selenium_scraper import SeleniumStocktwitsScraper
                print(f"  API failed, trying Selenium scraper for {symbol}...")
                selenium_scraper = SeleniumStocktwitsScraper(headless=True)
                messages_df = selenium_scraper.get_historical_messages(symbol, days_back)
                selenium_scraper.close()
                
                if len(messages_df) > 0:
                    sample = messages_df.iloc[0]['body'] if len(messages_df) > 0 else ''
                    if 'Message about' not in str(sample):
                        print(f"  Selenium scraper got {len(messages_df)} real messages")
                        return messages_df
            except ImportError:
                print(f"  Selenium scraper not available (install: pip install selenium webdriver-manager)")
            except Exception as e:
                print(f"  Selenium scraper failed: {e}")
            
            # Try web scraper as fallback
            if SCRAPER_AVAILABLE:
                print(f"  Trying web scraper for {symbol}...")
                try:
                    scraper = StocktwitsScraper()
                    messages_df = scraper.get_historical_messages(symbol, days_back)
                    if len(messages_df) > 0:
                        sample = messages_df.iloc[0]['body'] if len(messages_df) > 0 else ''
                        if 'Message about' not in str(sample):
                            print(f"  Web scraper got {len(messages_df)} real messages")
                            return messages_df
                except Exception as e:
                    print(f"  Web scraper failed: {e}")
            
            # Try historical data loader
            try:
                from historical_data_loader import HistoricalDataLoader
                loader = HistoricalDataLoader()
                hist_files = [
                    f"data/historical/{symbol}_messages.csv",
                    f"data/historical/stocktwits_{symbol}.csv",
                ]
                for filepath in hist_files:
                    if os.path.exists(filepath):
                        df = loader.load_from_file(filepath, symbol)
                        if len(df) > 0:
                            print(f"  Loaded {len(df)} messages from historical file")
                            return df
            except Exception as e:
                pass
            
            print(f"  No real data available for {symbol}")
            print(f"  ACTION REQUIRED: Download historical dataset or set up live scraping")
            raise ValueError(f"REAL DATA REQUIRED: No real Stocktwits data found for {symbol}. Synthetic data is disabled.")
        
        df = pd.DataFrame(messages)
        df = df.sort_values('timestamp')
        
        # Cache the results
        self._save_to_cache(symbol, days_back, df)
        
        print(f"  Collected {len(df)} real messages for {symbol}")
        return df
    
    def _parse_sentiment(self, sentiment_dict: Dict, message_body: str = "") -> float:
        """
        Enhanced sentiment parsing with NLP support
        """
        # 1. Get Stocktwits built-in sentiment
        base_sentiment = 0.0
        if sentiment_dict:
            basic = sentiment_dict.get('basic', '')
            if basic == 'bullish':
                base_sentiment = 1.0
            elif basic == 'bearish':
                base_sentiment = -1.0
        
        # 2. Add NLP-based sentiment if enabled
        if self.use_nlp and self.nlp_analyzer and message_body:
            try:
                nlp_sentiment = self.nlp_analyzer.analyze(message_body)
                # Weighted combination: 60% Stocktwits, 40% NLP
                combined = 0.6 * base_sentiment + 0.4 * nlp_sentiment
                return np.clip(combined, -1.0, 1.0)
            except Exception as e:
                # Fallback to base sentiment if NLP fails
                return base_sentiment
        
        return base_sentiment
    
    def _generate_synthetic_social_data(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Generate synthetic data as fallback"""
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        base_volume = np.random.randint(5, 50)
        volumes = base_volume + np.random.poisson(10, len(dates))
        sentiments = np.random.normal(0.1, 0.3, len(dates))
        sentiments = np.clip(sentiments, -1, 1)
        
        messages = []
        for date, vol, sent in zip(dates, volumes, sentiments):
            for _ in range(int(vol)):
                messages.append({
                    'timestamp': date + timedelta(
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    'id': np.random.randint(1000000, 9999999),
                    'body': f"Message about {symbol}",
                    'sentiment': sent,
                    'user_id': np.random.randint(1, 10000)
                })
        
        return pd.DataFrame(messages)
    
    def aggregate_daily_features(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate messages into daily features"""
        if len(messages_df) == 0:
            return pd.DataFrame()
        
        messages_df['date'] = pd.to_datetime(messages_df['timestamp']).dt.date
        
        daily = messages_df.groupby('date').agg({
            'id': 'count',
            'sentiment': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'volume', 'sentiment']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Calculate velocity
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

