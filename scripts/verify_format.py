"""
Verify that S_t (social sequences) are in the correct format: R^(T x d_s)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_processor import StreamBDataProcessor
from src.config import Config

def verify_s_t_format():
    """Verify S_t format matches R^(T x d_s)"""
    processor = StreamBDataProcessor()
    
    # Check a sample ticker
    import os
    tickers = [f.replace('_stocktwits.csv', '') for f in os.listdir('data/raw') if f.endswith('_stocktwits.csv')]
    
    if len(tickers) == 0:
        print("No tickers found. Run data collection first.")
        return False
    
    symbol = tickers[0]
    print(f"Verifying format for {symbol}...")
    
    data = processor.process_ticker(symbol)
    if data is None:
        print(f"No data for {symbol}")
        return False
    
    X, y = data
    
    # Verify format
    T = Config.SEQUENCE_LENGTH  # Should be 60
    d_s = Config.SOCIAL_FEATURE_DIM  # Should be 3
    
    print(f"\nSequence shape: {X.shape}")
    print(f"Expected: [N, T={T}, d_s={d_s}]")
    print(f"Actual: [N={X.shape[0]}, T={X.shape[1]}, d_s={X.shape[2]}]")
    
    # Verify dimensions
    assert X.shape[1] == T, f"Sequence length mismatch: expected {T}, got {X.shape[1]}"
    assert X.shape[2] == d_s, f"Feature dimension mismatch: expected {d_s}, got {X.shape[2]}"
    
    print(f"\n[OK] Format verification passed!")
    print(f"  - Each sample S_t has shape: [{T}, {d_s}] = R^({T} x {d_s})")
    print(f"  - Batch shape: [B, {T}, {d_s}]")
    print(f"  - Total samples: {X.shape[0]}")
    
    return True

if __name__ == "__main__":
    verify_s_t_format()

