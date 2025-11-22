"""
Data processing pipeline for Stream B
Combines Stocktwits social data with market data to create labels
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional
import os
from config import Config
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle

class StreamBDataProcessor:
    """Processes data for Stream B (Social Encoder)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.sequence_length = Config.SEQUENCE_LENGTH
        
    def get_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if len(df) == 0:
                return pd.DataFrame()
            
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Date': 'date'})
            
            # Calculate additional features
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=5).std()
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            return df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                      'returns', 'volatility', 'volume_ratio']]
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_labels(self, market_df: pd.DataFrame, 
                     surge_threshold: float = Config.SURGE_THRESHOLD_PCT,
                     forward_window: int = Config.SURGE_FORWARD_WINDOW,
                     min_volume_mult: float = Config.MIN_VOLUME_MULTIPLIER) -> pd.Series:
        """
        Create binary labels: 1 if surge occurs within forward_window sessions
        
        Surge definition:
        - Price increase >= surge_threshold%
        - Volume >= min_volume_mult * average volume
        """
        labels = pd.Series(0, index=market_df.index)
        
        for i in range(len(market_df) - forward_window):
            current_price = market_df.iloc[i]['Close']
            current_vol = market_df.iloc[i]['Volume']
            avg_vol = market_df.iloc[i:max(i+1, i+20)]['Volume'].mean()
            
            # Look forward
            future_prices = market_df.iloc[i+1:i+forward_window+1]['Close']
            max_future_price = future_prices.max()
            
            price_increase_pct = ((max_future_price - current_price) / current_price) * 100
            
            # Check surge conditions
            if price_increase_pct >= surge_threshold and current_vol >= min_volume_mult * avg_vol:
                labels.iloc[i] = 1
        
        return labels
    
    def align_data(self, social_df: pd.DataFrame, market_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align social and market data by date"""
        # Normalize timezones - remove timezone info for merging
        market_df = market_df.copy()
        social_df = social_df.copy()
        
        # Convert to datetime and remove timezone if present
        market_df['date'] = pd.to_datetime(market_df['date'])
        try:
            # If timezone-aware, convert to naive
            if market_df['date'].dt.tz is not None:
                market_df['date'] = market_df['date'].dt.tz_convert(None)
        except (AttributeError, TypeError):
            # Already timezone-naive or no tz attribute
            pass
        
        # Normalize to date only (remove time component)
        market_df['date'] = market_df['date'].dt.normalize()
        
        social_df['date'] = pd.to_datetime(social_df['date']).dt.normalize()
        
        # Merge on date
        merged = pd.merge(
            market_df,
            social_df,
            on='date',
            how='left'
        )
        
        # Forward fill missing social data (assume no change if no messages)
        merged[['volume', 'sentiment', 'velocity']] = merged[['volume', 'sentiment', 'velocity']].fillna(0)
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        return merged, merged[['volume', 'sentiment', 'velocity']]
    
    def create_sequences(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of length T for training
        
        Returns:
            X: [N, T, feature_dim] sequences
            y: [N] labels
        """
        sequences = []
        sequence_labels = []
        
        for i in range(len(features_df) - self.sequence_length):
            seq = features_df.iloc[i:i+self.sequence_length].values
            label = labels.iloc[i + self.sequence_length - 1]  # Label at end of sequence
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(sequence_labels, dtype=np.float32)
        
        return X, y
    
    def process_ticker(self, symbol: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process a single ticker: load social data, fetch market data, create sequences
        """
        # Load social data
        social_path = os.path.join(Config.RAW_DATA_DIR, f"{symbol}_stocktwits.csv")
        if not os.path.exists(social_path):
            print(f"Social data not found for {symbol}")
            return None
        
        social_df = pd.read_csv(social_path)
        social_df['date'] = pd.to_datetime(social_df['date'])
        
        # Determine date range
        if start_date is None:
            start_date = social_df['date'].min() - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = social_df['date'].max()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Get market data
        start_str = start_date.date() if hasattr(start_date, 'date') else str(start_date)[:10]
        end_str = end_date.date() if hasattr(end_date, 'date') else str(end_date)[:10]
        market_df = self.get_market_data(symbol, start_str, end_str)
        if len(market_df) == 0:
            print(f"No market data for {symbol}")
            return None
        
        # Align data
        aligned_df, social_features = self.align_data(social_df, market_df)
        
        # Create labels
        labels = self.create_labels(aligned_df)
        
        # Create sequences
        X, y = self.create_sequences(social_features, labels)
        
        if len(X) == 0:
            print(f"No valid sequences for {symbol}")
            return None
        
        return X, y
    
    def prepare_dataset(self, symbols: list, 
                       train_start: str, train_end: str,
                       val_start: str, val_end: str,
                       test_start: str, test_end: str) -> dict:
        """
        Prepare train/val/test datasets from multiple tickers
        """
        datasets = {
            'train': {'X': [], 'y': []},
            'val': {'X': [], 'y': []},
            'test': {'X': [], 'y': []}
        }
        
        for symbol in symbols:
            # Train data
            train_data = self.process_ticker(symbol, train_start, train_end)
            if train_data:
                X_train, y_train = train_data
                datasets['train']['X'].append(X_train)
                datasets['train']['y'].append(y_train)
            
            # Val data
            val_data = self.process_ticker(symbol, val_start, val_end)
            if val_data:
                X_val, y_val = val_data
                datasets['val']['X'].append(X_val)
                datasets['val']['y'].append(y_val)
            
            # Test data
            test_data = self.process_ticker(symbol, test_start, test_end)
            if test_data:
                X_test, y_test = test_data
                datasets['test']['X'].append(X_test)
                datasets['test']['y'].append(y_test)
        
        # Concatenate all tickers
        for split in ['train', 'val', 'test']:
            if len(datasets[split]['X']) > 0:
                datasets[split]['X'] = np.concatenate(datasets[split]['X'], axis=0)
                datasets[split]['y'] = np.concatenate(datasets[split]['y'], axis=0)
            else:
                datasets[split]['X'] = np.array([])
                datasets[split]['y'] = np.array([])
        
        # Fit scaler on training data
        if len(datasets['train']['X']) > 0:
            train_flat = datasets['train']['X'].reshape(-1, datasets['train']['X'].shape[-1])
            self.scaler.fit(train_flat)
            
            # Scale all splits
            for split in ['train', 'val', 'test']:
                if len(datasets[split]['X']) > 0:
                    shape = datasets[split]['X'].shape
                    flat = datasets[split]['X'].reshape(-1, shape[-1])
                    scaled = self.scaler.transform(flat)
                    datasets[split]['X'] = scaled.reshape(shape)
        
        return datasets
    
    def save_scaler(self, path: str):
        """Save fitted scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

