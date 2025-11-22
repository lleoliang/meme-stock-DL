"""
Main entry point for Meme Stock Surge Prediction (Stream B)
Run the complete pipeline: data collection, training, and backtesting
"""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.prepare_and_train import main as train_main
from src.backtest.backtest import Backtester
from tests.test_backtest import test_backtest_17_tickers
import torch
from src.models.stream_b import StreamBClassifier
from src.data.data_processor import StreamBDataProcessor
from src.config import Config

def main():
    parser = argparse.ArgumentParser(
        description='Meme Stock Surge Prediction - Stream B (Social Encoder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (data collection, training, backtesting)
  python main.py --mode full

  # Train only (assumes data already collected)
  python main.py --mode train

  # Backtest only (assumes model already trained)
  python main.py --mode backtest

  # Collect data only
  python main.py --mode collect
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'train', 'backtest', 'collect'],
        help='Execution mode: full (default), train, backtest, or collect'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='Specific symbols to process (default: use trending tickers)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='src/models/stream_b_best.pth',
        help='Path to trained model (for backtest mode)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MEME STOCK SURGE PREDICTION - STREAM B")
    print("="*70)
    print(f"Mode: {args.mode}")
    print("="*70)
    
    if args.mode == 'full':
        # Run complete pipeline
        train_main()
    
    elif args.mode == 'train':
        # Training only
        train_main()
    
    elif args.mode == 'backtest':
        # Backtesting only
        print("\nRunning backtest on existing model...")
        test_backtest_17_tickers()
    
    elif args.mode == 'collect':
        # Data collection only
        from src.data.data_collector_enhanced import EnhancedStocktwitsCollector
        collector = EnhancedStocktwitsCollector()
        
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = collector.get_trending_tickers(limit=50)
        
        print(f"\nCollecting data for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                collector.collect_ticker_data(symbol, save=True)
            except Exception as e:
                print(f"  Error collecting {symbol}: {e}")
                continue
        
        print("\nData collection complete!")
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

