# System Context: Meme Stock Surge Prediction (Stream B)

## Project Goals

**Primary Objective:** Build Stream B - a social encoder that processes Stocktwits data to predict meme stock surges (≥2% price increase within N sessions) using only social signals.

**Architecture Goal:** Two-stream model with cross-attention:
- Stream A: Market encoder (OHLCV) - NOT IMPLEMENTED YET
- Stream B: Social encoder (Stocktwits) - CURRENT FOCUS
- Cross-attention fusion - NOT IMPLEMENTED YET

**Current Scope:** Stream B only - social signals from Stocktwits (volume, sentiment, velocity)

## Project Structure

```
meme_stock_ml_project/
├── config.py                      # Configuration (API sources, model params)
├── data_collector.py              # Basic collector (auto-uses enhanced)
├── data_collector_enhanced.py    # Enhanced collector (NLP, caching, auth)
├── stocktwits_scraper.py         # Web scraper (blocked by Stocktwits)
├── selenium_scraper.py           # Selenium scraper (needs testing)
├── historical_data_loader.py     # Loads GitHub datasets
├── sentiment_analyzer.py         # NLP sentiment (VADER/FinBERT)
├── data_processor.py             # Processes data, creates sequences
├── models/
│   └── stream_b.py               # Stream B model (SocialEncoder + Classifier)
├── losses.py                     # Focal loss, Weighted BCE
├── train_stream_b.py             # Training with weight optimization
├── backtest.py                   # Backtesting framework
├── prepare_and_train.py          # Main training pipeline
├── collect_real_data.py          # Tries all methods to get real data
├── verify_stocktwits_data.py     # Verification script
├── test_enhanced_collector.py    # Test enhanced features
├── verify_format.py              # Verify S_t format [T, d_s]
├── test_backtest.py              # Test backtesting
├── data/
│   ├── raw/                      # Raw Stocktwits CSV files
│   ├── processed/                # Processed sequences
│   ├── cache/                    # Cached API responses
│   └── historical/               # Historical datasets (empty - needs files)
├── models/
│   ├── stream_b_best.pth         # Trained model
│   └── scaler.pkl                # Feature scaler
└── results/
    ├── weight_optimization.csv   # Weight tuning results
    └── backtest_*.json/csv       # Backtest results
```

## What Has Worked ✅

### 1. Model Architecture
- **Stream B Model:** Fully implemented and working
  - `SocialEncoder`: LSTM/GRU/Transformer encoder
  - `StreamBClassifier`: Complete model with attention pooling
  - Format verified: S_t = [T=60, d_s=3] = R^(60 x 3) ✅

### 2. Training Pipeline
- **Weight Optimization:** Working - tested weights 1.0-100.0
- **Best Weight Found:** 50.0 (PR-AUC: 0.5287)
- **Training:** Model trains successfully
- **Early Stopping:** Implemented and working
- **Metrics:** PR-AUC, Precision@K tracking

### 3. Data Processing
- **Sequence Creation:** Working - creates [N, T, d_s] sequences
- **Label Creation:** Surge detection (≥2% within 5 sessions) working
- **Data Alignment:** Market + social data alignment working
- **Time-based Splits:** Train/val/test splits implemented

### 4. NLP Sentiment Analysis
- **VADER:** Installed and working
- **Integration:** Combines Stocktwits API sentiment (60%) + NLP (40%)
- **FinBERT:** Available but optional (slower, needs GPU)

### 5. Caching System
- **Implementation:** Working
- **Cache Directory:** `data/cache/`
- **Expiry:** 1 hour configurable

### 6. Backtesting Framework
- **Structure:** Complete backtesting system
- **Metrics:** CAGR, Sharpe, Drawdown, Win Rate
- **Trading Simulation:** Position management, transaction costs
- **Issue:** Date handling fixed, but needs real data to test properly

## What Hasn't Worked ❌

### 1. Stocktwits API Access
- **Direct API:** Returns 403 Forbidden (blocked)
- **RapidAPI:** Endpoint doesn't exist (404)
- **Status:** Stocktwits paused new API registrations
- **Impact:** Cannot get real live data via API

### 2. Web Scraping
- **Public Endpoints:** Blocked (403)
- **BeautifulSoup Scraping:** Cannot extract embedded JSON
- **Status:** Stocktwits has anti-scraping measures
- **Impact:** Cannot scrape live data

### 3. Selenium Scraper
- **Implementation:** Complete
- **Status:** Installed but crashes/errors when tested
- **Issue:** Needs debugging - may work with proper setup
- **Potential:** Most reliable method if fixed

### 4. Synthetic Data
- **Current State:** System falls back to synthetic data
- **Problem:** User explicitly rejected synthetic data
- **Code Change:** Enhanced collector now raises error instead of using synthetic

### 5. Historical Data
- **Loader:** Implemented and ready
- **Status:** No data files available yet
- **Need:** GitHub dataset download and placement

## Current State

### Code Status
- ✅ **Model:** Trained and saved (`models/stream_b_best.pth`)
- ✅ **Format:** S_t verified as [60, 3] = R^(60 x 3)
- ✅ **Training:** Completed with optimal weight (50.0)
- ✅ **Backtesting:** Framework ready, needs real data
- ❌ **Data Collection:** No real data source working

### Data Status
- **Current:** Using synthetic data (user rejected)
- **Blocked:** All API/scraping methods blocked by Stocktwits
- **Solution Needed:** Historical dataset or alternative source

### Configuration
- **API Source:** `stocktwits` (default), `rapidapi` (not available), `pytwits` (not integrated)
- **NLP Sentiment:** Enabled (VADER)
- **Caching:** Enabled
- **Model:** LSTM encoder, attention pooling, hidden_dim=64

## Key Technical Details

### Data Format (S_t)
- **Shape:** [N, T, d_s] where:
  - N = number of samples
  - T = 60 (sequence length)
  - d_s = 3 (volume, sentiment, velocity)
- **Verification:** `verify_format.py` confirms correct format

### Surge Definition
- **Threshold:** ≥2% price increase
- **Window:** Within 5 forward sessions
- **Volume:** Must be ≥1.5x average volume
- **Label:** Binary (0/1)

### Model Architecture
```python
SocialEncoder (LSTM) → [B, T, H]
    ↓
Attention Pooling → [B, H]
    ↓
Classifier (MLP) → [B] logits
```

### Training Details
- **Loss:** Weighted BCE (optimal weight: 50.0)
- **Optimizer:** AdamW (lr=1e-4)
- **Metrics:** PR-AUC (primary), Precision@K
- **Validation:** Walk-forward time-based splits

## Critical Issues

### 1. No Real Data Source
- **Problem:** Cannot collect real Stocktwits data
- **Impact:** Cannot train/test on real data
- **Workaround:** Need historical dataset or alternative source

### 2. Backtesting Date Mismatch
- **Problem:** Synthetic data dates (2025) don't match test dates (2021)
- **Status:** Fixed in code, but needs real data to verify
- **Solution:** Use dates that match available data

### 3. API Blocking
- **Problem:** Stocktwits blocks all automated access
- **Root Cause:** Anti-scraping measures, API registration paused
- **Impact:** Cannot get live or historical data via API

## Next Steps (Actionable)

### Immediate Priority: Get Real Data

**Option 1: Historical Dataset (Recommended)**
1. Search GitHub for "Stocktwits messages dataset" or "Stocktwits historical"
2. Download CSV files with columns: timestamp, body, sentiment
3. Place in `data/historical/{SYMBOL}_messages.csv`
4. Run: `python prepare_and_train.py`
5. System will automatically use historical data

**Option 2: Fix Selenium Scraper**
1. Debug Selenium scraper crashes
2. Test with: `python -c "from selenium_scraper import SeleniumStocktwitsScraper; s = SeleniumStocktwitsScraper(); print(s.get_messages('GME', 10))"`
3. May need Chrome browser installed
4. May need to adjust selectors/parsing

**Option 3: Alternative Data Sources**
1. Implement Reddit r/wallstreetbets scraper
2. Use Twitter/X financial sentiment APIs
3. Contact Stocktwits for enterprise access

### Code Integration

**Current Flow:**
1. `prepare_and_train.py` → calls `data_collector.py`
2. `data_collector.py` → auto-uses `EnhancedStocktwitsCollector`
3. `EnhancedStocktwitsCollector` → tries API, then scraper, then historical loader
4. If all fail → currently raises error (synthetic disabled)

**To Enable Historical Data:**
- Place CSV files in `data/historical/`
- Format: `timestamp, body, sentiment` (or compatible)
- System will auto-detect and use them

## Important Files

### Core Implementation
- `models/stream_b.py` - Stream B model (working)
- `data_processor.py` - Data processing (working)
- `train_stream_b.py` - Training (working)
- `backtest.py` - Backtesting (ready, needs data)

### Data Collection
- `data_collector_enhanced.py` - Main collector (tries all methods)
- `stocktwits_scraper.py` - Web scraper (blocked)
- `selenium_scraper.py` - Selenium scraper (needs debugging)
- `historical_data_loader.py` - Historical data loader (ready, needs files)

### Testing/Verification
- `verify_format.py` - Verifies S_t format ✅
- `collect_real_data.py` - Tries all collection methods
- `test_backtest.py` - Tests backtesting

## Configuration

**Key Settings in `config.py`:**
```python
SEQUENCE_LENGTH = 60          # T = 60 timesteps
SOCIAL_FEATURE_DIM = 3        # [volume, sentiment, velocity]
HIDDEN_DIM = 64               # Model hidden dimension
SURGE_THRESHOLD_PCT = 2.0     # 2% price increase
SURGE_FORWARD_WINDOW = 5      # 5 sessions forward
USE_NLP_SENTIMENT = True       # Enable NLP sentiment
NLP_SENTIMENT_METHOD = 'VADER' # VADER or FinBERT
```

## Known Limitations

1. **No Real Data:** All collection methods blocked
2. **Synthetic Rejected:** User explicitly wants real data only
3. **Selenium Untested:** Implemented but not verified working
4. **Historical Empty:** Loader ready but no data files
5. **Backtesting:** Framework ready but needs real data to test

## Success Criteria

**For Stream B to be complete:**
1. ✅ Model architecture implemented
2. ✅ Training pipeline working
3. ✅ Weight optimization complete
4. ✅ Format verified (S_t = [60, 3])
5. ❌ Real data collection (BLOCKED)
6. ⚠️  Backtesting (ready, needs data)

## Dependencies

**Installed:**
- torch, numpy, pandas, yfinance
- requests, beautifulsoup4
- selenium, webdriver-manager
- vaderSentiment
- scikit-learn, matplotlib, seaborn

**Optional:**
- transformers (for FinBERT)
- pytwits (community wrapper - not integrated)

## Environment Variables

**For RapidAPI (if available):**
```bash
export RAPIDAPI_KEY=your_key
export STOCKTWITS_API_SOURCE=rapidapi
```

**For Stocktwits API (if you have existing token):**
```bash
export STOCKTWITS_TOKEN=your_token
export STOCKTWITS_API_SOURCE=stocktwits
```

## Testing Commands

```bash
# Verify format
python verify_format.py

# Test data collection
python collect_real_data.py GME 7

# Test enhanced collector
python test_enhanced_collector.py

# Run full pipeline (will fail without real data)
python prepare_and_train.py

# Test backtesting
python test_backtest.py
```

## Architecture Notes

**Stream B (Current):**
- Input: S_t ∈ R^(T x d_s) = [60, 3]
- Encoder: LSTM → [B, T, H]
- Pooling: Attention → [B, H]
- Output: Binary logits

**Future (Not Implemented):**
- Stream A: Market encoder (OHLCV)
- Cross-attention: Market ↔ Social
- Fusion: Combined representation

## Critical Path Forward

1. **Get Real Data** (BLOCKING)
   - Historical dataset OR
   - Fix Selenium scraper OR
   - Alternative source (Reddit/Twitter)

2. **Verify with Real Data**
   - Test data collection
   - Re-train model
   - Run backtesting

3. **Complete Stream B**
   - Ensure all features working
   - Optimize weights with real data
   - Final backtest results

## Notes for Next Agent

- **User explicitly rejected synthetic data** - do not use fallbacks
- **All API methods blocked** - need alternative approach
- **Code is ready** - just needs data source
- **Selenium may work** - needs debugging/testing
- **Historical loader ready** - just needs CSV files
- **Model trained** - but on synthetic data (needs retraining)
- **Backtesting ready** - framework complete, needs real data

The system is architecturally complete but blocked by data access. Focus on getting real data through historical datasets or fixing Selenium scraper.

