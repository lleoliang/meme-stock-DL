# Deployment Guide

This guide explains how to deploy and run the Meme Stock Surge Prediction system.

## Prerequisites

1. **Python 3.8+**
2. **Chrome Browser** (for Selenium scraper)
3. **CUDA-capable GPU** (optional, for faster training)

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd meme-stock-DL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from selenium import webdriver; print('Selenium OK')"
```

## Data Setup

### Option 1: Use Existing Data

If you have CSV files in `data/raw/`, the system will use them automatically.

### Option 2: Collect New Data

```bash
# Collect data for specific symbols
python main.py --mode collect --symbols GME TSLA AMC

# Or collect for trending tickers
python main.py --mode collect
```

### Option 3: Use Historical Data

1. Download historical Stocktwits datasets
2. Place CSV files in `data/historical/`
3. Format: `timestamp,body,sentiment` or `date,volume,sentiment,velocity`

## Running the System

### Full Pipeline

```bash
python main.py --mode full
```

This runs:
1. Data collection
2. Data processing
3. Weight optimization
4. Model training
5. Backtesting

### Individual Steps

```bash
# Step 1: Collect data
python main.py --mode collect

# Step 2: Train model
python main.py --mode train

# Step 3: Backtest
python main.py --mode backtest
```

## Configuration

Edit `config.py` to customize:

```python
# Model parameters
SEQUENCE_LENGTH = 60          # Time window
HIDDEN_DIM = 64               # Model size
SURGE_THRESHOLD_PCT = 2.0     # 2% surge threshold

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Backtesting
TOP_K_PREDICTIONS = 10
INITIAL_CAPITAL = 100000
TRANSACTION_COST_PCT = 0.001
```

## Environment Variables (Optional)

```bash
# For Stocktwits API (if you have access)
export STOCKTWITS_TOKEN=your_token
export STOCKTWITS_API_SOURCE=stocktwits

# For RapidAPI (if available)
export RAPIDAPI_KEY=your_key
export STOCKTWITS_API_SOURCE=rapidapi
```

## Output Files

After running, check:

- `models/stream_b_best.pth`: Trained model
- `models/scaler.pkl`: Feature scaler
- `results/weight_optimization.csv`: Weight tuning results
- `results/backtest_results.json`: Backtest metrics
- `results/backtest_trades.csv`: Trade log
- `results/backtest_plots.png`: Performance plots

## Troubleshooting

### No Data Available

If data collection fails:
1. Check `data/raw/` for existing CSV files
2. Use historical data loader
3. Manually collect data and format as CSV

### Model Training Fails

1. Ensure sufficient data (min 30 days per ticker)
2. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reduce batch size if memory issues

### Backtesting Errors

1. Ensure model is trained: `ls models/stream_b_best.pth`
2. Check data dates match backtest period
3. Verify scaler exists: `ls models/scaler.pkl`

## Production Deployment

For production use:

1. **Model Serving**: Use Flask/FastAPI to serve predictions
2. **Data Pipeline**: Set up scheduled data collection
3. **Monitoring**: Track model performance over time
4. **Retraining**: Schedule periodic retraining with new data

Example API endpoint:

```python
from flask import Flask, request, jsonify
import torch
from models.stream_b import StreamBClassifier

app = Flask(__name__)
model = load_model('models/stream_b_best.pth')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process and predict
    prediction = model.predict(data)
    return jsonify({'prediction': prediction})
```

## Support

For issues or questions:
1. Check `SYSTEM_CONTEXT.md` for detailed system information
2. Review error messages and logs
3. Ensure all dependencies are installed correctly

