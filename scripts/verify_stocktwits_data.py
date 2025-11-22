"""
Verify if we're getting real Stocktwits data or synthetic data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_collector import StocktwitsCollector
import pandas as pd
import os

def verify_stocktwits_data():
    """Check if we're getting real Stocktwits API data"""
    print("="*70)
    print("VERIFYING STOCKTWITS DATA COLLECTION")
    print("="*70)
    
    collector = StocktwitsCollector()
    
    # Test with a few tickers
    test_tickers = ['GME', 'AAPL', 'TSLA']
    
    for symbol in test_tickers:
        print(f"\n{'='*70}")
        print(f"Testing {symbol}")
        print(f"{'='*70}")
        
        # Get raw messages
        messages_df = collector.get_stocktwits_messages(symbol, days_back=7)
        
        print(f"Total messages collected: {len(messages_df)}")
        
        if len(messages_df) == 0:
            print("[WARNING] No messages - API may have failed")
            continue
        
        # Check if it's synthetic data
        sample_bodies = messages_df['body'].head(5).tolist()
        is_synthetic = all('Message about' in str(body) for body in sample_bodies)
        
        if is_synthetic:
            print("[WARNING] USING SYNTHETIC DATA - API not returning real data")
            print(f"Sample messages: {sample_bodies[:3]}")
        else:
            print("[OK] REAL STOCKTWITS DATA DETECTED")
            print(f"Sample messages:")
            for i, body in enumerate(sample_bodies[:3], 1):
                print(f"  {i}. {body[:100]}...")
        
        # Check sentiment distribution
        if len(messages_df) > 0:
            sentiment_counts = messages_df['sentiment'].value_counts()
            print(f"\nSentiment distribution:")
            print(f"  Bullish (1.0): {sentiment_counts.get(1.0, 0)}")
            print(f"  Bearish (-1.0): {sentiment_counts.get(-1.0, 0)}")
            print(f"  Neutral (0.0): {sentiment_counts.get(0.0, 0)}")
            print(f"  Average sentiment: {messages_df['sentiment'].mean():.3f}")
        
        # Check date range
        if len(messages_df) > 0:
            print(f"\nDate range:")
            print(f"  From: {messages_df['timestamp'].min()}")
            print(f"  To: {messages_df['timestamp'].max()}")
        
        # Check saved data
        saved_path = os.path.join('data', 'raw', f'{symbol}_stocktwits.csv')
        if os.path.exists(saved_path):
            saved_df = pd.read_csv(saved_path)
            print(f"\nSaved data: {len(saved_df)} daily records")
            print(f"  Date range: {saved_df['date'].min()} to {saved_df['date'].max()}")
            print(f"  Features: volume, sentiment, velocity")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\nRecommendations:")
    print("1. If using synthetic data: Add API authentication or check API status")
    print("2. If getting real data: Consider adding NLP sentiment analysis")
    print("3. Check rate limits: Stocktwits allows 200 requests/hour")

if __name__ == "__main__":
    verify_stocktwits_data()

