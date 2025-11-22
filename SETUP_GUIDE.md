# Stocktwits Enhanced Collection Setup Guide

## ✅ All 4 Improvements Implemented

### 1. ✅ Verification Script
**File**: `verify_stocktwits_data.py`
- Checks if we're getting real Stocktwits data or synthetic
- Shows sentiment distribution and date ranges
- Run: `python verify_stocktwits_data.py`

### 2. ✅ NLP-Based Sentiment Analysis
**File**: `sentiment_analyzer.py`
- **VADER**: Fast, no GPU needed (default)
- **FinBERT**: More accurate, requires transformers library
- Automatically combines Stocktwits sentiment (60%) with NLP sentiment (40%)
- Configure in `config.py`: `USE_NLP_SENTIMENT = True`, `NLP_SENTIMENT_METHOD = 'VADER'`

### 3. ✅ API Authentication Support
**File**: `data_collector_enhanced.py`
- Set environment variable: `STOCKTWITS_TOKEN=your_token_here`
- Or add to `config.py`: `STOCKTWITS_ACCESS_TOKEN = 'your_token'`
- Get token from: https://stocktwits.com/developers
- Benefits: Higher rate limits (30 requests vs 10), better access

### 4. ✅ Caching & Rate Limiting
**File**: `data_collector_enhanced.py`
- Automatic caching of API responses
- Cache expiry: 1 hour (configurable)
- Better rate limit handling (waits on 429 errors)
- Cache location: `data/cache/`

## Quick Start

### Install Dependencies
```bash
pip install vaderSentiment
# Optional for FinBERT:
pip install transformers torch
```

### Set API Token (Optional but Recommended)
```bash
# Windows PowerShell
$env:STOCKTWITS_TOKEN="your_token_here"

# Linux/Mac
export STOCKTWITS_TOKEN="your_token_here"
```

### Verify Setup
```bash
python verify_stocktwits_data.py
python test_enhanced_collector.py
```

## Configuration

Edit `config.py`:

```python
# Enable NLP sentiment
USE_NLP_SENTIMENT = True
NLP_SENTIMENT_METHOD = 'VADER'  # or 'FinBERT'

# Enable caching
CACHE_SOCIAL_DATA = True
CACHE_EXPIRY_HOURS = 1

# API token (or set via environment variable)
STOCKTWITS_ACCESS_TOKEN = os.environ.get('STOCKTWITS_TOKEN', '')
```

## Usage

### Basic Usage (Auto-upgrades to enhanced)
```python
from data_collector import StocktwitsCollector

collector = StocktwitsCollector()  # Automatically uses enhanced if available
data = collector.collect_ticker_data('GME')
```

### Direct Enhanced Usage
```python
from data_collector_enhanced import EnhancedStocktwitsCollector

collector = EnhancedStocktwitsCollector()
data = collector.collect_ticker_data('GME')
```

## Features Comparison

| Feature | Basic Collector | Enhanced Collector |
|---------|----------------|-------------------|
| Stocktwits API | ✅ | ✅ |
| Built-in Sentiment | ✅ | ✅ |
| NLP Sentiment | ❌ | ✅ (VADER/FinBERT) |
| API Authentication | ❌ | ✅ |
| Caching | ❌ | ✅ |
| Rate Limit Handling | Basic | Advanced |
| Synthetic Fallback | ✅ | ✅ |

## Current Status

**Verification Results:**
- Currently using **SYNTHETIC DATA** (API not returning real data)
- This is expected if:
  - No API token set
  - Rate limits exceeded
  - API temporarily unavailable

**To Get Real Data:**
1. Get Stocktwits API token from https://stocktwits.com/developers
2. Set environment variable: `STOCKTWITS_TOKEN=your_token`
3. Run collection again

## Troubleshooting

### "VADER not installed"
```bash
pip install vaderSentiment
```

### "No real data collected"
- Check if API token is set: `echo $STOCKTWITS_TOKEN`
- Check rate limits (200 requests/hour for public API)
- API may be temporarily unavailable

### "Authentication failed"
- Verify your API token is correct
- Check token hasn't expired
- Ensure token has proper permissions

### Cache Issues
- Clear cache: `rm -rf data/cache/` (Linux/Mac) or `rmdir /s data\cache` (Windows)
- Adjust cache expiry in `config.py`

## Next Steps

1. **Get API Token**: Sign up at https://stocktwits.com/developers
2. **Set Token**: `export STOCKTWITS_TOKEN="your_token"`
3. **Test**: `python test_enhanced_collector.py`
4. **Collect Data**: Run `prepare_and_train.py` to collect real data

## Files Created

- ✅ `verify_stocktwits_data.py` - Verification script
- ✅ `sentiment_analyzer.py` - NLP sentiment analysis
- ✅ `data_collector_enhanced.py` - Enhanced collector with all features
- ✅ `test_enhanced_collector.py` - Test script
- ✅ `SETUP_GUIDE.md` - This guide
- ✅ Updated `config.py` - Added configuration options
- ✅ Updated `requirements.txt` - Added vaderSentiment

