# Deployment Checklist

## ✅ Files Ready for Deployment

### Core Application Files
- [x] `main.py` - Main entry point with CLI
- [x] `config.py` - Configuration parameters
- [x] `prepare_and_train.py` - Training pipeline
- [x] `backtest.py` - Backtesting framework
- [x] `data_processor.py` - Data preprocessing
- [x] `train_stream_b.py` - Training utilities
- [x] `losses.py` - Loss functions

### Model Files
- [x] `models/stream_b.py` - Model architecture
- [x] `models/__init__.py` - Package init

### Data Collection
- [x] `data_collector_enhanced.py` - Enhanced collector
- [x] `data_collector.py` - Basic collector
- [x] `selenium_scraper.py` - Selenium scraper
- [x] `sentiment_analyzer.py` - NLP sentiment
- [x] `historical_data_loader.py` - Historical data loader

### Documentation
- [x] `README.md` - Main documentation
- [x] `DEPLOYMENT.md` - Deployment guide
- [x] `SYSTEM_CONTEXT.md` - System context
- [x] `.gitignore` - Git ignore rules
- [x] `requirements.txt` - Dependencies

### Testing (Optional)
- [x] `test_backtest.py` - Backtest tests
- [ ] `test_*.py` - Other test files (can be excluded)

## 📦 Files to Exclude from Deployment

These should be in `.gitignore`:
- `__pycache__/` - Python cache
- `*.pth`, `*.pkl` - Model files (too large)
- `data/raw/*.csv` - Raw data files (use sample only)
- `results/*` - Results (regenerated)
- `test_*.py` - Test files (except test_backtest.py)
- `.vscode/`, `.idea/` - IDE files

## 🚀 Deployment Steps

### 1. Prepare Repository

```bash
# Clean up test files
rm test_chrome_install.py test_selenium_scraper.py test_stocktwits_access.py
rm test_file.txt test_write.txt

# Keep only essential test
# test_backtest.py is useful, keep it
```

### 2. Add Sample Data (Optional)

Create a sample data file to show format:

```bash
# Create sample data directory structure
mkdir -p data/raw data/processed data/cache data/historical
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/cache/.gitkeep
touch data/historical/.gitkeep
```

### 3. Create Sample Data File

Add one sample CSV file to show the expected format:
- `data/raw/GME_sample.csv` (first 10 rows)

### 4. Verify Structure

```bash
# Check all core files exist
ls main.py config.py prepare_and_train.py backtest.py
ls models/stream_b.py data_processor.py train_stream_b.py
ls data_collector_enhanced.py sentiment_analyzer.py
```

### 5. Test Installation

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt

# Test imports
python -c "from models.stream_b import StreamBClassifier; print('OK')"
python -c "from data_processor import StreamBDataProcessor; print('OK')"
```

## 📋 Pre-Deployment Checklist

- [ ] All code files are clean and commented
- [ ] README.md is up to date
- [ ] requirements.txt includes all dependencies
- [ ] .gitignore excludes unnecessary files
- [ ] Sample data file included (if needed)
- [ ] Configuration is documented
- [ ] Entry point (main.py) works
- [ ] No hardcoded paths (use config.py)
- [ ] Error handling is robust
- [ ] Logging is appropriate

## 🎯 Post-Deployment

After deployment, users should be able to:

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py --mode full`
4. Or use individual modes: `python main.py --mode train`

## 📝 Notes

- Model files (`*.pth`, `*.pkl`) are excluded due to size
- Users will need to train their own models or download separately
- Data files are excluded; users provide their own
- Test files are mostly excluded except `test_backtest.py`

