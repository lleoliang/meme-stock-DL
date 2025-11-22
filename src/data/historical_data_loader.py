"""
Load historical Stocktwits data from GitHub datasets
This provides real historical data for training
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
from pathlib import Path
import requests
import zipfile
import io

from src.config import Config
from src.utils.sentiment_analyzer import SentimentAnalyzer

class HistoricalDataLoader:
    """Load historical Stocktwits data from GitHub or local files"""
    
    def __init__(self):
        self.data_dir = Path('data/historical')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub datasets (examples - user can add their own)
        self.github_datasets = [
            # Add GitHub repo URLs here when found
            # Example: "https://github.com/user/repo/raw/main/stocktwits_data.csv"
        ]
        
        # Initialize NLP sentiment
        self.nlp_analyzer = None
        if Config.USE_NLP_SENTIMENT:
            try:
                self.nlp_analyzer = SentimentAnalyzer(method=Config.NLP_SENTIMENT_METHOD)
            except:
                pass
    
    def load_from_file(self, filepath: str, symbol: str = None) -> pd.DataFrame:
        """Load historical data from a CSV/JSON file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix == '.json':
                df = pd.read_json(filepath)
            else:
                print(f"Unsupported file format: {filepath.suffix}")
                return pd.DataFrame()
            
            # Process the data
            return self._process_historical_data(df, symbol)
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
    
    def _process_historical_data(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Process historical data into our format"""
        # Try to identify columns
        # Common formats: timestamp, date, created_at, body, message, sentiment
        
        messages = []
        
        # Find timestamp column
        time_cols = ['timestamp', 'created_at', 'date', 'time', 'datetime']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        # Find body/message column
        body_cols = ['body', 'message', 'text', 'content']
        body_col = None
        for col in body_cols:
            if col in df.columns:
                body_col = col
                break
        
        # Find sentiment column
        sent_cols = ['sentiment', 'sentiment_score', 'polarity']
        sent_col = None
        for col in sent_cols:
            if col in df.columns:
                sent_col = col
                break
        
        if not time_col or not body_col:
            print("Could not identify required columns in historical data")
            return pd.DataFrame()
        
        # Process each row
        for _, row in df.iterrows():
            try:
                timestamp = pd.to_datetime(row[time_col])
                body = str(row[body_col])
                
                # Get sentiment
                if sent_col and pd.notna(row[sent_col]):
                    sentiment = float(row[sent_col])
                else:
                    # Extract sentiment from message
                    sentiment = self._extract_sentiment_from_text(body)
                
                messages.append({
                    'timestamp': timestamp,
                    'id': row.get('id', hash(body) % 10000000),
                    'body': body,
                    'sentiment': sentiment,
                    'user_id': row.get('user_id', 0)
                })
            except Exception as e:
                continue
        
        if len(messages) == 0:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(messages)
        result_df = result_df.sort_values('timestamp')
        
        # Filter by symbol if provided
        if symbol:
            # Check if body contains symbol
            result_df = result_df[result_df['body'].str.contains(symbol, case=False, na=False)]
        
        return result_df
    
    def _extract_sentiment_from_text(self, text: str) -> float:
        """Extract sentiment from text using NLP"""
        if self.nlp_analyzer:
            try:
                return self.nlp_analyzer.analyze(text)
            except:
                return 0.0
        return 0.0
    
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

