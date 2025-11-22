"""
Test backtesting on the 17 collected tickers
"""
import torch
import pandas as pd
from datetime import datetime, timedelta
import os

from src.models.stream_b import StreamBClassifier
from src.data.data_processor import StreamBDataProcessor
from src.backtest.backtest import Backtester
from src.config import Config
import pickle

def test_backtest_17_tickers():
    """Run backtest on the 17 collected tickers"""
    print("="*70)
    print("TESTING BACKTEST ON 17 TICKERS")
    print("="*70)
    
    # Get list of tickers
    tickers = []
    for f in os.listdir('data/raw'):
        if f.endswith('_stocktwits.csv'):
            ticker = f.replace('_stocktwits.csv', '')
            tickers.append(ticker)
    
    print(f"\nFound {len(tickers)} tickers: {tickers}")
    
    if len(tickers) == 0:
        print("No tickers found. Please run data collection first.")
        return
    
    # Load model
    print("\nLoading trained model...")
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('models/stream_b_best.pth', map_location=device, weights_only=False)
    model = StreamBClassifier(
        input_dim=Config.SOCIAL_FEATURE_DIM,
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout'],
        encoder_type='LSTM',
        use_attention=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded. Optimal weight: {checkpoint['config']['optimal_weight']}")
    
    # Load processor with scaler
    processor = StreamBDataProcessor()
    processor.load_scaler('models/scaler.pkl')
    
    # Create backtester
    backtester = Backtester(model, processor, device=device)
    
    # Define test period - use dates that match the synthetic data
    # Synthetic data is generated from today going back 365 days
    # So we'll use recent dates that should be in the data
    from datetime import datetime, timedelta
    test_end = datetime.now() - timedelta(days=30)  # 30 days ago
    test_start = test_end - timedelta(days=60)  # 60 days before that
    
    # Get business days
    test_dates = pd.date_range(start=test_start, end=test_end, freq='D')
    test_dates = [d for d in test_dates if d.weekday() < 5]  # Business days only
    
    print(f"\nTest period: {test_start.date()} to {test_end.date()}")
    print(f"Number of trading days: {len(test_dates)}")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = backtester.get_predictions(tickers, test_dates)
    
    print(f"\nPredictions generated for {len(predictions)} tickers")
    total_predictions = sum(len(df) for df in predictions.values())
    print(f"Total prediction points: {total_predictions}")
    
    if len(predictions) == 0:
        print("No predictions generated. Check data availability.")
        return
    
    # Run backtest
    print("\n" + "="*70)
    print("RUNNING BACKTEST")
    print("="*70)
    
    backtest_results = backtester.simulate_trading(predictions, top_k=Config.TOP_K_PREDICTIONS)
    
    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"CAGR: {backtest_results['cagr']*100:.2f}%")
    print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"\nNumber of Trades: {backtest_results['num_trades']}")
    print(f"Win Rate: {backtest_results['win_rate']*100:.2f}%")
    if backtest_results['num_trades'] > 0:
        print(f"Average Win: {backtest_results['avg_win']*100:.2f}%")
        print(f"Average Loss: {backtest_results['avg_loss']*100:.2f}%")
    
    # Save results
    import json
    results_path = 'results/test_backtest_results.json'
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            k: v for k, v in backtest_results.items() 
            if k not in ['portfolio_value', 'trades', 'daily_returns']
        }, f, indent=2, default=str)
    
    # Save trades
    if len(backtest_results['trades']) > 0:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_path = 'results/test_backtest_trades.csv'
        trades_df.to_csv(trades_path, index=False)
        print(f"\nSaved trades to {trades_path}")
        print(f"\nFirst 5 trades:")
        print(trades_df.head().to_string())
    
    # Plot results
    plot_path = 'results/test_backtest_plots.png'
    backtester.plot_results(backtest_results, save_path=plot_path)
    print(f"\nSaved plots to {plot_path}")
    
    print("\n" + "="*70)
    print("BACKTEST TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_backtest_17_tickers()

