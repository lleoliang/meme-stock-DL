"""
Backtesting framework for Stream B predictions
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from src.models.stream_b import StreamBClassifier
from src.config import Config
from src.data.data_processor import StreamBDataProcessor
import yfinance as yf

class Backtester:
    """
    Backtesting framework for Stream B model
    Simulates trading based on model predictions
    """
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor
        self.device = device
        self.initial_capital = Config.INITIAL_CAPITAL
        self.transaction_cost = Config.TRANSACTION_COST_PCT
        
    def get_predictions(self, symbols: List[str], dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """
        Get model predictions for all symbols on given dates
        Pre-processes all data once, then extracts sequences for each date
        
        Returns:
            Dict mapping symbol to DataFrame with columns: date, prediction, probability
        """
        predictions = {}
        
        # Pre-process all tickers once
        print("Pre-processing ticker data...")
        ticker_data = {}
        for symbol in symbols:
            try:
                # Process entire ticker history
                data = self.processor.process_ticker(symbol)
                if data is not None and len(data[0]) > 0:
                    ticker_data[symbol] = data
                    print(f"  Loaded {symbol}: {len(data[0])} sequences")
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
                continue
        
        if len(ticker_data) == 0:
            print("No ticker data loaded for predictions")
            return predictions
        
        # Now get predictions for each date
        print(f"\nGenerating predictions for {len(dates)} dates...")
        for symbol in ticker_data.keys():
            symbol_preds = []
            X, y = ticker_data[symbol]
            
            # We need to map sequences to dates
            # Load the aligned data to get dates
            try:
                social_path = os.path.join(Config.RAW_DATA_DIR, f"{symbol}_stocktwits.csv")
                if not os.path.exists(social_path):
                    continue
                
                social_df = pd.read_csv(social_path)
                social_df['date'] = pd.to_datetime(social_df['date'])
                
                # Get market data to align dates
                market_df = self.processor.get_market_data(
                    symbol,
                    str(social_df['date'].min().date()),
                    str(social_df['date'].max().date())
                )
                
                if len(market_df) == 0:
                    continue
                
                # Align data to get dates
                aligned_df, _ = self.processor.align_data(social_df, market_df)
                
                # Create date mapping for sequences
                # Each sequence ends at index i + sequence_length - 1
                sequence_dates = []
                seq_len = Config.SEQUENCE_LENGTH
                
                # Ensure we have enough data
                if len(aligned_df) < seq_len:
                    continue
                
                # Create sequence dates - match exactly with how sequences were created
                for i in range(len(aligned_df) - seq_len):
                    seq_end_idx = i + seq_len - 1
                    seq_date = aligned_df.iloc[seq_end_idx]['date']
                    # Normalize date to remove time component
                    if isinstance(seq_date, pd.Timestamp):
                        seq_date = seq_date.normalize()
                    sequence_dates.append(seq_date)
                
                # Match sequence count - use minimum to ensure alignment
                min_len = min(len(sequence_dates), len(X))
                sequence_dates = sequence_dates[:min_len]
                X = X[:min_len]
                
                if min_len == 0:
                    continue
                
                # Create date to sequence index mapping
                date_to_seq_idx = {}
                for idx, seq_date in enumerate(sequence_dates):
                    date_normalized = pd.to_datetime(seq_date).normalize()
                    date_to_seq_idx[date_normalized] = idx
                
                # Debug: Check date ranges
                if len(sequence_dates) > 0:
                    min_seq_date = min(pd.to_datetime(sequence_dates))
                    max_seq_date = max(pd.to_datetime(sequence_dates))
                    min_test_date = min(dates)
                    max_test_date = max(dates)
                    
                    # Only process if dates overlap
                    if max_seq_date < min_test_date or min_seq_date > max_test_date:
                        print(f"  Skipping {symbol}: sequence dates ({min_seq_date.date()} to {max_seq_date.date()}) don't overlap with test dates ({min_test_date.date()} to {max_test_date.date()})")
                        continue
                
                # Get predictions for requested dates
                for date in dates:
                    # Normalize date for comparison
                    date_normalized = pd.to_datetime(date).normalize()
                    
                    # Find closest sequence date (on or before this date)
                    valid_dates = [d for d in date_to_seq_idx.keys() if d <= date_normalized]
                    if len(valid_dates) == 0:
                        continue
                    
                    closest_date = max(valid_dates)
                    seq_idx = date_to_seq_idx[closest_date]
                    
                    if seq_idx >= len(X):
                        continue
                    
                    # Get sequence and predict
                    X_seq = torch.FloatTensor(X[seq_idx:seq_idx+1]).to(self.device)
                    
                    with torch.no_grad():
                        logits = self.model(X_seq)
                        prob = torch.sigmoid(logits).cpu().item()
                    
                    symbol_preds.append({
                        'date': date,
                        'symbol': symbol,
                        'prediction': 1 if prob > 0.5 else 0,
                        'probability': prob
                    })
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
            
            if len(symbol_preds) > 0:
                predictions[symbol] = pd.DataFrame(symbol_preds)
                print(f"  Generated {len(symbol_preds)} predictions for {symbol}")
        
        return predictions
    
    def simulate_trading(self, predictions: Dict[str, pd.DataFrame],
                        top_k: int = Config.TOP_K_PREDICTIONS) -> Dict:
        """
        Simulate trading strategy:
        - Each day, select top K stocks by prediction probability
        - Buy equal-weighted positions
        - Hold for forward_window days
        - Track returns, drawdowns, etc.
        """
        portfolio_value = [self.initial_capital]
        positions = {}  # symbol -> (entry_date, entry_price, shares)
        trades = []
        daily_returns = []
        
        # Get all unique dates
        all_dates = set()
        for df in predictions.values():
            all_dates.update(df['date'].tolist())
        all_dates = sorted(all_dates)
        
        for date in all_dates:
            # Get predictions for this date
            daily_preds = []
            for symbol, df in predictions.items():
                day_data = df[df['date'] == date]
                if len(day_data) > 0:
                    daily_preds.append({
                        'symbol': symbol,
                        'probability': day_data.iloc[0]['probability'],
                        'prediction': day_data.iloc[0]['prediction']
                    })
            
            if len(daily_preds) == 0:
                continue
            
            # Sort by probability and take top K
            daily_preds.sort(key=lambda x: x['probability'], reverse=True)
            top_picks = [p for p in daily_preds[:top_k] if p['prediction'] == 1]
            
            # Close positions that have been held for forward_window days
            positions_to_close = []
            for symbol, (entry_date, entry_price, shares) in positions.items():
                days_held = (date - entry_date).days
                if days_held >= Config.SURGE_FORWARD_WINDOW:
                    positions_to_close.append(symbol)
            
            # Calculate current cash (from previous day's portfolio value minus open positions)
            # We need to track cash separately
            if len(portfolio_value) == 1:
                current_cash = self.initial_capital
            else:
                # Estimate cash from previous portfolio value
                prev_positions_value = 0
                for sym, (ed, ep, sh) in positions.items():
                    try:
                        ticker = yf.Ticker(sym)
                        hist = ticker.history(start=str((date - pd.Timedelta(days=5)).date()), 
                                             end=str(date.date()))
                        if len(hist) > 0:
                            prev_positions_value += sh * hist['Close'].iloc[-1]
                        else:
                            prev_positions_value += sh * ep
                    except:
                        prev_positions_value += sh * ep
                current_cash = portfolio_value[-1] - prev_positions_value
            
            for symbol in positions_to_close:
                entry_date, entry_price, shares = positions.pop(symbol)
                
                # Get exit price - use historical date, not current date
                try:
                    ticker = yf.Ticker(symbol)
                    # Fetch historical data for the exit date
                    hist = ticker.history(start=str((date - pd.Timedelta(days=2)).date()), 
                                         end=str((date + pd.Timedelta(days=1)).date()))
                    if len(hist) > 0:
                        # Get the price on or before the exit date
                        exit_price = hist['Close'].iloc[-1]
                    else:
                        exit_price = entry_price
                except Exception as e:
                    exit_price = entry_price
                
                # Calculate return
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - 2 * self.transaction_cost  # Buy + sell
                pnl = shares * entry_price * net_return
                
                current_cash += shares * exit_price * (1 - self.transaction_cost)
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': net_return,
                    'pnl': pnl,
                    'days_held': days_held
                })
            
            # Open new positions
            capital_per_position = current_cash / len(top_picks) if len(top_picks) > 0 else 0
            
            for pick in top_picks:
                symbol = pick['symbol']
                
                # Skip if already in positions
                if symbol in positions:
                    continue
                
                try:
                    ticker = yf.Ticker(symbol)
                    # Use historical date for entry price
                    hist = ticker.history(start=str((date - pd.Timedelta(days=2)).date()), 
                                         end=str((date + pd.Timedelta(days=1)).date()))
                    if len(hist) > 0:
                        entry_price = hist['Close'].iloc[-1]
                    else:
                        continue
                except:
                    continue
                
                shares = (capital_per_position * (1 - self.transaction_cost)) / entry_price
                positions[symbol] = (date, entry_price, shares)
                current_cash -= shares * entry_price * (1 + self.transaction_cost)
            
            # Calculate portfolio value at end of day
            position_value = 0
            for symbol, (entry_date, entry_price, shares) in positions.items():
                try:
                    ticker = yf.Ticker(symbol)
                    # Use historical date for current price
                    hist = ticker.history(start=str((date - pd.Timedelta(days=2)).date()), 
                                         end=str((date + pd.Timedelta(days=1)).date()))
                    if len(hist) > 0:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        current_price = entry_price
                except:
                    current_price = entry_price
                
                position_value += shares * current_price
            
            total_value = current_cash + position_value
            portfolio_value.append(total_value)
            
            # Daily return
            if len(portfolio_value) > 1:
                daily_return = (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2]
                daily_returns.append(daily_return)
        
        # Close remaining positions
        final_date = all_dates[-1] if all_dates else datetime.now()
        for symbol, (entry_date, entry_price, shares) in positions.items():
            try:
                ticker = yf.Ticker(symbol)
                # Use historical date range
                hist = ticker.history(start=str((final_date - pd.Timedelta(days=2)).date()), 
                                     end=str((final_date + pd.Timedelta(days=1)).date()))
                if len(hist) > 0:
                    exit_price = hist['Close'].iloc[-1]
                else:
                    exit_price = entry_price
            except:
                exit_price = entry_price
            
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * self.transaction_cost
            pnl = shares * entry_price * net_return
            
            trades.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'exit_date': final_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': net_return,
                'pnl': pnl,
                'days_held': (final_date - entry_date).days
            })
        
        # Calculate metrics
        final_value = portfolio_value[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # CAGR
        days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 1 else 1
        years = days / 365.25
        cagr = (final_value / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_value:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Sharpe ratio (simplified)
        if len(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Win rate
        if len(trades) > 0:
            winning_trades = [t for t in trades if t['return'] > 0]
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['return'] for t in trades if t['return'] <= 0]) if any(t['return'] <= 0 for t in trades) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_value': portfolio_value,
            'trades': trades,
            'daily_returns': daily_returns
        }
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(results['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Value ($)')
        axes[0, 0].grid(True)
        
        # Returns distribution
        if len(results['daily_returns']) > 0:
            axes[0, 1].hist(results['daily_returns'], bins=50, edgecolor='black')
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Trade returns
        if len(results['trades']) > 0:
            trade_returns = [t['return'] for t in results['trades']]
            axes[1, 0].bar(range(len(trade_returns)), trade_returns)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_title('Individual Trade Returns')
            axes[1, 0].set_xlabel('Trade')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].grid(True)
        
        # Metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""
        Backtest Results
        
        Initial Capital: ${results['initial_capital']:,.2f}
        Final Value: ${results['final_value']:,.2f}
        Total Return: {results['total_return']*100:.2f}%
        CAGR: {results['cagr']*100:.2f}%
        Max Drawdown: {results['max_drawdown']*100:.2f}%
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        
        Number of Trades: {results['num_trades']}
        Win Rate: {results['win_rate']*100:.2f}%
        Avg Win: {results['avg_win']*100:.2f}%
        Avg Loss: {results['avg_loss']*100:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

