# Meme Stock Surge Prediction - Stream B (Social Encoder)

A deep learning model that predicts meme stock surges (≥2% price increase within 5 sessions) using social signals from Stocktwits. The model uses LSTM/Transformer encoders with attention mechanisms to learn patterns from social volume, sentiment, and velocity.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd meme-stock-DL

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete pipeline (data collection, training, backtesting)
python main.py --mode full

# Or run individual steps
python main.py --mode collect    # Collect Stocktwits data
python main.py --mode train      # Train model
python main.py --mode backtest   # Run backtesting
```

## 📁 Project Structure

```
meme-stock-DL/
├── main.py                      # Main entry point (redirects to src.main)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── main.py                 # Main CLI entry point
│   ├── config.py               # Configuration parameters
│   │
│   ├── models/                 # Model architecture
│   │   ├── __init__.py
│   │   └── stream_b.py         # Stream B model
│   │
│   ├── data/                   # Data collection & processing
│   │   ├── __init__.py
│   │   ├── data_collector.py
│   │   ├── data_collector_enhanced.py
│   │   ├── data_processor.py
│   │   ├── historical_data_loader.py
│   │   └── scrapers/
│   │       ├── selenium_scraper.py
│   │       └── stocktwits_scraper.py
│   │
│   ├── training/               # Training modules
│   │   ├── __init__.py
│   │   ├── prepare_and_train.py
│   │   ├── train_stream_b.py
│   │   └── losses.py
│   │
│   ├── backtest/               # Backtesting
│   │   ├── __init__.py
│   │   └── backtest.py
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── sentiment_analyzer.py
│
├── scripts/                     # Utility scripts
│   ├── collect_real_data.py
│   ├── quick_setup.py
│   └── verify_format.py
│
├── tests/                       # Test files
│   └── test_backtest.py
│
├── docs/                        # Documentation
│   ├── DEPLOYMENT.md
│   ├── QUICKSTART.md
│   └── SYSTEM_CONTEXT.md
│
├── data/                        # Data files
│   ├── raw/                    # Raw CSV files
│   └── samples/                # Sample data
│
└── examples/                   # Example code
    ├── model.py
    └── stock_data_scraper.py
```

## 🎯 Features

- **Social Encoder**: LSTM/GRU/Transformer encoder for Stocktwits signals
- **Multi-source Data Collection**: API, web scraping, and historical data support
- **NLP Sentiment Analysis**: VADER and FinBERT integration
- **Weight Optimization**: Automated hyperparameter tuning for class imbalance
- **Backtesting**: Full trading simulation with transaction costs
- **Comprehensive Metrics**: PR-AUC, Precision@K, CAGR, Sharpe Ratio, Max Drawdown

## 📊 Model Architecture

**Stream B** consists of:

1. **Social Encoder**: Processes sequences of [volume, sentiment, velocity] with shape [N, 60, 3]
2. **Attention Pooling**: Self-attention mechanism for sequence aggregation
3. **Classifier**: Multi-layer MLP for binary surge prediction

```
Input: S_t ∈ R^(60 x 3) = [volume, sentiment, velocity]
  ↓
LSTM Encoder → [B, 60, H]
  ↓
Attention Pooling → [B, H]
  ↓
MLP Classifier → [B] logits
```

## ⚙️ Configuration

Edit `config.py` to adjust:

- `SEQUENCE_LENGTH`: Time window size (default: 60)
- `HIDDEN_DIM`: Model hidden dimension (default: 64)
- `SURGE_THRESHOLD_PCT`: Price increase threshold (default: 2%)
- `SURGE_FORWARD_WINDOW`: Prediction horizon (default: 5 sessions)
- `TOP_K_PREDICTIONS`: Number of stocks to trade (default: 10)

## 📈 Data Collection

The system supports multiple data collection methods:

1. **Stocktwits API** (requires authentication)
2. **Web Scraping** (Selenium-based)
3. **Historical Data** (CSV files)

Data is automatically aggregated into daily features:
- **Volume**: Message count per day
- **Sentiment**: Average sentiment score (-1 to 1)
- **Velocity**: Rate of change in message volume

## 🏋️ Training

The training pipeline includes:

- **Class Imbalance Handling**: Weighted BCE loss with optimized weights
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stabilizes training
- **Metrics**: PR-AUC (primary), Precision@K

```bash
python prepare_and_train.py
```

This will:
1. Collect/load Stocktwits data
2. Process and align with market data
3. Optimize class weights via grid search
4. Train the final model with optimal weights
5. Run backtesting

## 📉 Backtesting

The backtesting framework simulates trading:

- **Strategy**: Top-K stocks by prediction probability
- **Position Management**: Equal-weighted, hold for forward_window days
- **Transaction Costs**: 0.1% per trade
- **Metrics**: CAGR, Sharpe Ratio, Max Drawdown, Win Rate

```bash
python main.py --mode backtest
```

Results are saved to `results/`:
- `backtest_results.json`: Performance metrics
- `backtest_trades.csv`: Individual trade log
- `backtest_plots.png`: Visualization

## 🔧 Advanced Usage

### Custom Symbols

```bash
python main.py --mode collect --symbols GME TSLA AMC
```

### Load Pre-trained Model

```python
import torch
from models.stream_b import StreamBClassifier
from config import Config

checkpoint = torch.load('models/stream_b_best.pth')
model = StreamBClassifier(
    input_dim=Config.SOCIAL_FEATURE_DIM,
    hidden_dim=checkpoint['config']['hidden_dim'],
    num_layers=checkpoint['config']['num_layers'],
    dropout=checkpoint['config']['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Data Collection Methods

The enhanced collector tries multiple methods in order:
1. Stocktwits API (if authenticated)
2. Selenium scraper
3. Historical data loader
4. Raises error (synthetic data disabled)

## 📝 Notes

- **Academic Use Only**: This is for research/backtesting purposes
- **API Limitations**: Stocktwits API has rate limits and may require authentication
- **Data Quality**: Ensure sufficient data points per ticker (min: 30 days)
- **Stocktwits Blocking**: Stocktwits actively blocks automated access; use historical data or manual collection

## 🐛 Troubleshooting

### Stocktwits API Issues

If you encounter 403 errors:
- Stocktwits has paused new API registrations
- Use historical data files in `data/historical/`
- Try Selenium scraper (may be blocked)
- Consider alternative data sources (Reddit, Twitter)

### Selenium Issues

If Selenium fails:
- Ensure Chrome browser is installed
- Install: `pip install selenium webdriver-manager`
- Try non-headless mode: Set `headless=False` in `selenium_scraper.py`

### Data Issues

If no data is found:
- Check `data/raw/` directory for CSV files
- Ensure CSV format: `date,volume,sentiment,velocity`
- Use `collect_real_data.py` to test data collection

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:

- `torch>=2.0.0`: Deep learning framework
- `pandas>=2.0.0`: Data manipulation
- `yfinance>=0.2.28`: Market data
- `selenium>=4.15.0`: Web scraping
- `vaderSentiment>=3.3.2`: Sentiment analysis

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guide
- Add tests for new features
- Update documentation

## 📄 License

This project is for educational/research purposes only.

## 🙏 Acknowledgments

- Stocktwits for social data
- Yahoo Finance for market data
- VADER Sentiment for NLP analysis
