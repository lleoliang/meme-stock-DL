# Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Check Your Setup

```bash
# Verify Python version (3.8+)
python --version

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify Chrome (for Selenium)
# Make sure Chrome browser is installed
```

## Step 3: Prepare Data

You have three options:

### Option A: Use Existing Data
If you have CSV files in `data/raw/`, you're ready!

### Option B: Collect Data
```bash
python main.py --mode collect --symbols GME TSLA
```

### Option C: Use Sample Data
The system includes sample data format. Add your own CSV files to `data/raw/` with format:
```csv
date,volume,sentiment,velocity
2024-01-01,1.2,0.5,0.1
2024-01-02,1.5,0.6,0.3
```

## Step 4: Run the Pipeline

```bash
# Full pipeline (recommended for first run)
python main.py --mode full
```

This will:
1. ✅ Load/collect data
2. ✅ Process and create sequences
3. ✅ Optimize weights
4. ✅ Train model
5. ✅ Run backtest

## Step 5: Check Results

```bash
# View results
ls results/

# Check model
ls models/
```

## Common Issues

### "No data found"
- Add CSV files to `data/raw/`
- Or run: `python main.py --mode collect`

### "Chrome driver error"
- Install Chrome browser
- Or skip Selenium: Use existing CSV files

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in `config.py`
- Or use CPU: Set `DEVICE = "cpu"` in `config.py`

## Next Steps

- Read `README.md` for detailed documentation
- Check `DEPLOYMENT.md` for advanced setup
- Review `SYSTEM_CONTEXT.md` for system architecture

## Getting Help

1. Check error messages carefully
2. Review `SYSTEM_CONTEXT.md` for known issues
3. Ensure all dependencies are installed
4. Verify data format matches expected structure

Happy trading! 📈

