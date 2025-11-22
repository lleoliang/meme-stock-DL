"""
Main script to prepare data and train Stream B with backtesting
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import json

# Use enhanced collector if available
try:
    from src.data.data_collector_enhanced import EnhancedStocktwitsCollector as StocktwitsCollector
except ImportError:
    from src.data.data_collector import StocktwitsCollector
from src.data.data_processor import StreamBDataProcessor
from src.training.train_stream_b import optimize_weights, StreamBTrainer, SocialDataset
from src.backtest.backtest import Backtester
from src.models.stream_b import StreamBClassifier
from src.training.losses import WeightedBCELoss
from src.config import Config
from torch.utils.data import DataLoader

def main():
    print("="*70)
    print("STREAM B: Social Encoder Training & Backtesting")
    print("="*70)
    
    # Set random seeds
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    
    # Create directories
    for dir_path in [Config.DATA_DIR, Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR,
                     Config.MODELS_DIR, Config.RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Step 1: Collect Stocktwits data
    print("\n" + "="*70)
    print("STEP 1: Collecting Stocktwits Data")
    print("="*70)
    
    collector = StocktwitsCollector()
    
    # Get tickers
    print("Fetching ticker list...")
    tickers = collector.get_trending_tickers(limit=Config.TICKER_LIMIT)
    print(f"Found {len(tickers)} tickers")
    
    # Collect data for each ticker
    print("\nCollecting Stocktwits data...")
    collected_tickers = []
    for i, ticker in enumerate(tickers[:50]):  # Limit to 50 for initial testing
        print(f"[{i+1}/{min(50, len(tickers))}] Processing {ticker}...")
        data = collector.collect_ticker_data(ticker, save=True)
        if data is not None and len(data) >= Config.MIN_SOCIAL_DATA_POINTS:
            collected_tickers.append(ticker)
        if (i + 1) % 10 == 0:
            print(f"  Collected data for {len(collected_tickers)} tickers so far...")
    
    print(f"\nSuccessfully collected data for {len(collected_tickers)} tickers")
    
    if len(collected_tickers) == 0:
        print("ERROR: No tickers with sufficient data. Exiting.")
        return
    
    # Step 2: Process data and create datasets
    print("\n" + "="*70)
    print("STEP 2: Processing Data and Creating Datasets")
    print("="*70)
    
    processor = StreamBDataProcessor()
    
    # Define time splits
    train_end = datetime(2020, 12, 31)
    val_end = datetime(2021, 6, 30)
    test_end = datetime(2022, 12, 31)
    
    train_start = train_end - timedelta(days=365)
    val_start = train_end
    test_start = val_end
    
    print(f"Train: {train_start.date()} to {train_end.date()}")
    print(f"Val: {val_start.date()} to {val_end.date()}")
    print(f"Test: {test_start.date()} to {test_end.date()}")
    
    print("\nProcessing tickers...")
    datasets = processor.prepare_dataset(
        collected_tickers,
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        val_start=str(val_start.date()),
        val_end=str(val_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date())
    )
    
    # Save scaler
    scaler_path = os.path.join(Config.MODELS_DIR, "scaler.pkl")
    processor.save_scaler(scaler_path)
    
    # Print dataset stats
    print("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        X = datasets[split]['X']
        y = datasets[split]['y']
        if len(X) > 0:
            pos_ratio = y.sum() / len(y)
            print(f"{split.upper()}: {len(X)} samples, "
                  f"positive ratio: {pos_ratio:.4f} ({y.sum()}/{len(y)})")
        else:
            print(f"{split.upper()}: No data")
    
    if len(datasets['train']['X']) == 0:
        print("ERROR: No training data. Exiting.")
        return
    
    # Step 3: Optimize weights
    print("\n" + "="*70)
    print("STEP 3: Optimizing Class Weights")
    print("="*70)
    
    weight_range = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    best_weight_result, weight_results_df = optimize_weights(
        datasets['train'],
        datasets['val'],
        weight_range=weight_range,
        use_focal_loss=False
    )
    
    # Save weight optimization results
    weight_results_path = os.path.join(Config.RESULTS_DIR, "weight_optimization.csv")
    weight_results_df.to_csv(weight_results_path, index=False)
    print(f"\nSaved weight optimization results to {weight_results_path}")
    
    optimal_weight = best_weight_result['weight']
    print(f"\nOptimal weight: {optimal_weight}")
    
    # Step 4: Train final model with optimal weight
    print("\n" + "="*70)
    print("STEP 4: Training Final Model with Optimal Weight")
    print("="*70)
    
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = StreamBClassifier(
        input_dim=Config.SOCIAL_FEATURE_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        encoder_type='LSTM',
        use_attention=True
    )
    
    criterion = WeightedBCELoss(pos_weight=optimal_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    train_dataset = SocialDataset(datasets['train']['X'], datasets['train']['y'])
    val_dataset = SocialDataset(datasets['val']['X'], datasets['val']['y'])
    test_dataset = SocialDataset(datasets['test']['X'], datasets['test']['y'])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    trainer = StreamBTrainer(model, device=device)
    best_pr_auc = trainer.train(
        train_loader, val_loader,
        criterion, optimizer,
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 5: Evaluating on Test Set")
    print("="*70)
    
    test_metrics = trainer.evaluate(test_loader, criterion)
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test Precision@K: {test_metrics['precision_at_k']:.4f}")
    
    # Save model
    model_path = os.path.join(Config.MODELS_DIR, "stream_b_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': Config.SOCIAL_FEATURE_DIM,
            'hidden_dim': Config.HIDDEN_DIM,
            'num_layers': Config.NUM_LAYERS,
            'dropout': Config.DROPOUT,
            'optimal_weight': optimal_weight
        },
        'test_metrics': test_metrics
    }, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Step 6: Backtesting
    print("\n" + "="*70)
    print("STEP 6: Running Backtest")
    print("="*70)
    
    # Load processor with scaler
    processor.load_scaler(scaler_path)
    
    backtester = Backtester(model, processor, device=device)
    
    # Get test period dates
    test_dates = pd.date_range(start=test_start, end=test_end, freq='D')
    test_dates = [d for d in test_dates if d.weekday() < 5]  # Business days only
    
    print(f"Running backtest on {len(test_dates)} dates...")
    print(f"Using top {Config.TOP_K_PREDICTIONS} predictions per day")
    
    predictions = backtester.get_predictions(collected_tickers[:20], test_dates[:30])  # Limit for testing
    
    if len(predictions) > 0:
        backtest_results = backtester.simulate_trading(predictions, top_k=Config.TOP_K_PREDICTIONS)
        
        # Print results
        print("\nBacktest Results:")
        print(f"  Initial Capital: ${backtest_results['initial_capital']:,.2f}")
        print(f"  Final Value: ${backtest_results['final_value']:,.2f}")
        print(f"  Total Return: {backtest_results['total_return']*100:.2f}%")
        print(f"  CAGR: {backtest_results['cagr']*100:.2f}%")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Number of Trades: {backtest_results['num_trades']}")
        print(f"  Win Rate: {backtest_results['win_rate']*100:.2f}%")
        
        # Save results
        results_path = os.path.join(Config.RESULTS_DIR, "backtest_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                k: v for k, v in backtest_results.items() 
                if k not in ['portfolio_value', 'trades', 'daily_returns']
            }, f, indent=2, default=str)
        
        # Save trades
        if len(backtest_results['trades']) > 0:
            trades_df = pd.DataFrame(backtest_results['trades'])
            trades_path = os.path.join(Config.RESULTS_DIR, "backtest_trades.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"\nSaved trades to {trades_path}")
        
        # Plot results
        plot_path = os.path.join(Config.RESULTS_DIR, "backtest_plots.png")
        backtester.plot_results(backtest_results, save_path=plot_path)
        print(f"Saved plots to {plot_path}")
    else:
        print("No predictions generated for backtesting")
    
    print("\n" + "="*70)
    print("STREAM B TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

