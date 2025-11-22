"""
Stocktwits Web Scraper - Uses public JSON endpoints
Gets real live and historical data without API keys
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import re
import os
from bs4 import BeautifulSoup

from src.config import Config
from src.utils.sentiment_analyzer import SentimentAnalyzer

class StocktwitsScraper:
    """Scrape Stocktwits using their public JSON endpoints"""
    
    def __init__(self):
        self.base_url = "https://stocktwits.com"
        self.api_base = "https://api.stocktwits.com/api/2"
        self.session = requests.Session()
        # Better headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://stocktwits.com/',
            'Origin': 'https://stocktwits.com',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        })
        self.rate_limit_delay = 1.5  # Be respectful
        
        # Initialize NLP sentiment
        self.nlp_analyzer = None
        if Config.USE_NLP_SENTIMENT:
            try:
                self.nlp_analyzer = SentimentAnalyzer(method=Config.NLP_SENTIMENT_METHOD)
            except:
                pass
    
    def get_symbol_stream(self, symbol: str, limit: int = 30) -> List[Dict]:
        """
        Get live stream for a symbol using public endpoint
        Try multiple methods to get real data
        """
        messages = []
        
        # Method 1: Try public API endpoint (may be blocked)
        url = f"{self.api_base}/streams/symbol/{symbol}.json"
        
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                if 'messages' in data:
                    messages.extend(data['messages'])
                    print(f"  Got {len(data['messages'])} messages from API endpoint")
                    return messages
            elif response.status_code == 403:
                print(f"  API blocked (403), trying alternative methods...")
            else:
                print(f"  API returned {response.status_code}")
        except Exception as e:
            print(f"  API error: {e}")
        
        # Method 2: Try web page scraping for embedded data
        if len(messages) == 0:
            messages = self._scrape_web_stream(symbol, limit)
        
        # Method 3: Try alternative endpoint format
        if len(messages) == 0:
            messages = self._try_alternative_endpoints(symbol)
        
        return messages
    
    def _try_alternative_endpoints(self, symbol: str) -> List[Dict]:
        """Try alternative API endpoint formats"""
        messages = []
        
        # Try without .json extension
        alternatives = [
            f"{self.api_base}/streams/symbol/{symbol}",
            f"{self.base_url}/api/2/streams/symbol/{symbol}.json",
        ]
        
        for url in alternatives:
            try:
                response = self.session.get(url, timeout=10)
                time.sleep(self.rate_limit_delay)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if 'messages' in data:
                            messages.extend(data['messages'])
                            print(f"  Got {len(data['messages'])} messages from alternative endpoint")
                            break
                    except:
                        pass
            except:
                continue
        
        return messages
    
    def _scrape_web_stream(self, symbol: str, limit: int = 30) -> List[Dict]:
        """
        Scrape Stocktwits web page - look for API calls the page makes
        """
        messages = []
        
        try:
            page_url = f"{self.base_url}/symbol/{symbol}"
            
            response = self.session.get(page_url, timeout=15)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                html = response.text
                
                # Method 1: Look for embedded JSON in script tags
                soup = BeautifulSoup(html, 'html.parser')
                scripts = soup.find_all('script')
                
                for script in scripts:
                    if not script.string:
                        continue
                    
                    script_text = script.string
                    
                    # Look for window.__INITIAL_STATE__ or similar
                    patterns = [
                        r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});',
                        r'window\.__APOLLO_STATE__\s*=\s*(\{.*?\});',
                        r'"messages"\s*:\s*\[(.*?)\]',
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, script_text, re.DOTALL)
                        for match in matches:
                            try:
                                # Try to extract and parse JSON
                                json_str = match.group(1) if match.groups() else match.group(0)
                                # Clean up the JSON string
                                json_str = json_str.strip().rstrip(';')
                                data = json.loads(json_str)
                                
                                # Navigate to messages
                                if isinstance(data, dict):
                                    if 'messages' in data:
                                        msgs = data['messages']
                                    elif 'stream' in data and 'messages' in data['stream']:
                                        msgs = data['stream']['messages']
                                    else:
                                        # Try to find messages recursively
                                        msgs = self._find_messages_in_dict(data)
                                    
                                    if msgs:
                                        messages.extend(msgs[:limit])
                                        print(f"  Found {len(msgs)} messages in embedded JSON")
                                        return messages[:limit]
                            except:
                                continue
                
                # Method 2: Look for API endpoint URLs in the page
                api_urls = re.findall(r'https://api\.stocktwits\.com/api/2/[^"\s]+', html)
                for api_url in set(api_urls[:3]):  # Try first 3 unique URLs
                    try:
                        resp = self.session.get(api_url, timeout=10)
                        time.sleep(self.rate_limit_delay)
                        if resp.status_code == 200:
                            data = resp.json()
                            if 'messages' in data:
                                messages.extend(data['messages'][:limit])
                                print(f"  Got {len(data['messages'])} messages from discovered endpoint")
                                return messages[:limit]
                    except:
                        continue
        
        except Exception as e:
            print(f"  Web scraping error: {e}")
        
        return messages
    
    def _find_messages_in_dict(self, data: dict, depth: int = 0) -> List:
        """Recursively find messages in nested dict"""
        if depth > 5:  # Limit recursion
            return []
        
        if isinstance(data, dict):
            if 'messages' in data and isinstance(data['messages'], list):
                return data['messages']
            for value in data.values():
                result = self._find_messages_in_dict(value, depth + 1)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_messages_in_dict(item, depth + 1)
                if result:
                    return result
        
        return []
    
    def get_historical_messages(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get historical messages by paginating through streams
        """
        all_messages = []
        max_id = None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"  Collecting historical data for {symbol} (last {days_back} days)...")
        
        # Try to get as many messages as possible
        for page in range(20):  # Limit to 20 pages
            try:
                url = f"{self.api_base}/streams/symbol/{symbol}.json"
                params = {}
                if max_id:
                    params['max'] = max_id
                
                response = self.session.get(url, params=params, timeout=10)
                time.sleep(self.rate_limit_delay)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                if 'messages' not in data or len(data['messages']) == 0:
                    break
                
                page_messages = []
                for msg in data['messages']:
                    try:
                        msg_date = datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        if msg_date < start_date:
                            # Reached start date, stop
                            break
                        
                        # Extract sentiment
                        sentiment = self._extract_sentiment(msg)
                        
                        page_messages.append({
                            'timestamp': msg_date,
                            'id': msg['id'],
                            'body': msg.get('body', ''),
                            'sentiment': sentiment,
                            'user_id': msg.get('user', {}).get('id', 0)
                        })
                    except Exception as e:
                        continue
                
                if len(page_messages) == 0:
                    break
                
                all_messages.extend(page_messages)
                
                # Check if we've gone back far enough
                oldest_date = min(m['timestamp'] for m in page_messages)
                if oldest_date < start_date:
                    break
                
                # Get next page cursor
                if 'cursor' in data and 'max' in data['cursor']:
                    max_id = data['cursor']['max']
                else:
                    break
                
                print(f"    Page {page+1}: {len(page_messages)} messages (total: {len(all_messages)})")
                
            except Exception as e:
                print(f"    Error on page {page+1}: {e}")
                break
        
        if len(all_messages) == 0:
            print(f"  No messages collected, trying live stream...")
            # Try getting at least current messages
            live_messages = self.get_symbol_stream(symbol, limit=50)
            for msg in live_messages:
                try:
                    msg_date = datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                    sentiment = self._extract_sentiment(msg)
                    all_messages.append({
                        'timestamp': msg_date,
                        'id': msg['id'],
                        'body': msg.get('body', ''),
                        'sentiment': sentiment,
                        'user_id': msg.get('user', {}).get('id', 0)
                    })
                except:
                    continue
        
        if len(all_messages) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_messages)
        df = df.sort_values('timestamp')
        print(f"  Collected {len(df)} real messages for {symbol}")
        
        return df
    
    def _extract_sentiment(self, msg: Dict) -> float:
        """Extract sentiment from message"""
        # Try Stocktwits built-in sentiment first
        sentiment_dict = msg.get('entities', {}).get('sentiment', {})
        base_sentiment = 0.0
        
        if sentiment_dict:
            basic = sentiment_dict.get('basic', '')
            if basic == 'bullish':
                base_sentiment = 1.0
            elif basic == 'bearish':
                base_sentiment = -1.0
        
        # Add NLP sentiment if available
        body = msg.get('body', '')
        if self.nlp_analyzer and body:
            try:
                nlp_sentiment = self.nlp_analyzer.analyze(body)
                combined = 0.6 * base_sentiment + 0.4 * nlp_sentiment
                return np.clip(combined, -1.0, 1.0)
            except:
                return base_sentiment
        
        return base_sentiment
    
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
    
    def collect_ticker_data(self, symbol: str, days_back: int = 365, save: bool = True) -> Optional[pd.DataFrame]:
        """Collect real Stocktwits data for a ticker"""
        print(f"Scraping Stocktwits data for {symbol}...")
        
        messages_df = self.get_historical_messages(symbol, days_back)
        
        if len(messages_df) == 0:
            print(f"  No data collected for {symbol}")
            return None
        
        daily_features = self.aggregate_daily_features(messages_df)
        
        if save:
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            filepath = os.path.join(Config.RAW_DATA_DIR, f"{symbol}_stocktwits.csv")
            daily_features.to_csv(filepath, index=False)
            print(f"  Saved {len(daily_features)} daily records to {filepath}")
        
        return daily_features

