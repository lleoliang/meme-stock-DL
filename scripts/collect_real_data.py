"""
Collect REAL Stocktwits data - Actionable script
Tries multiple methods to get real data
"""
import os
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.scrapers.stocktwits_scraper import StocktwitsScraper
from src.data.historical_data_loader import HistoricalDataLoader
from src.data.data_collector_enhanced import EnhancedStocktwitsCollector

# Try to import Selenium scraper
try:
    from src.data.scrapers.selenium_scraper import SeleniumStocktwitsScraper
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

def collect_real_data(symbol: str, days_back: int = 30):
    """
    Try all methods to get REAL Stocktwits data
    Returns real data or None
    """
    print(f"="*70)
    print(f"COLLECTING REAL DATA FOR {symbol}")
    print(f"="*70)
    
    # Method 1: Try web scraper
    print("\nMethod 1: Web Scraper")
    print("-" * 70)
    try:
        scraper = StocktwitsScraper()
        messages = scraper.get_historical_messages(symbol, days_back)
        if len(messages) > 0:
            sample = messages.iloc[0]['body']
            if 'Message about' not in sample:
                print(f"[OK] SUCCESS: Got {len(messages)} REAL messages")
                return messages
            else:
                print(f"[WARN] Got synthetic data")
        else:
            print(f"[FAIL] No messages from scraper")
    except Exception as e:
        print(f"[FAIL] Scraper error: {e}")
    
    # Method 2: Try enhanced collector (may use API if available)
    print("\nMethod 2: Enhanced Collector")
    print("-" * 70)
    try:
        collector = EnhancedStocktwitsCollector()
        messages = collector.get_stocktwits_messages(symbol, days_back)
        if len(messages) > 0:
            sample = messages.iloc[0]['body']
            if 'Message about' not in sample:
                print(f"[OK] SUCCESS: Got {len(messages)} REAL messages")
                return messages
            else:
                print(f"[WARN] Got synthetic data")
        else:
            print(f"[FAIL] No messages from collector")
    except Exception as e:
        print(f"[FAIL] Collector error: {e}")
    
    # Method 3: Try Selenium scraper (most reliable)
    if SELENIUM_AVAILABLE:
        print("\nMethod 3: Selenium Scraper")
        print("-" * 70)
        try:
            selenium_scraper = SeleniumStocktwitsScraper(headless=True)
            # Try get_historical_messages first (better for historical data)
            df = selenium_scraper.get_historical_messages(symbol, days_back)
            
            if len(df) == 0:
                # Fallback to get_messages
                selenium_messages = selenium_scraper.get_messages(symbol, limit=100)
                if len(selenium_messages) > 0:
                    df = pd.DataFrame(selenium_messages)
            
            selenium_scraper.close()
            
            if len(df) > 0:
                if 'body' in df.columns:
                    sample = str(df.iloc[0]['body']) if len(df) > 0 else ''
                    if 'Message about' not in sample and len(sample) > 10:
                        print(f"[OK] SUCCESS: Got {len(df)} REAL messages via Selenium")
                        return df
        except Exception as e:
            print(f"[FAIL] Selenium error: {e}")
            import traceback
            traceback.print_exc()
            print("  Install: pip install selenium webdriver-manager")
    
    # Method 4: Check for historical data files
    print("\nMethod 4: Historical Data Files")
    print("-" * 70)
    historical_loader = HistoricalDataLoader()
    
    # Check common locations
    historical_files = [
        f"data/historical/{symbol}_messages.csv",
        f"data/historical/stocktwits_{symbol}.csv",
        f"data/historical/{symbol}.csv",
    ]
    
    for filepath in historical_files:
        if os.path.exists(filepath):
            print(f"  Found historical file: {filepath}")
            df = historical_loader.load_from_file(filepath, symbol)
            if len(df) > 0:
                print(f"[OK] SUCCESS: Loaded {len(df)} messages from file")
                return df
    
    print("\n" + "="*70)
    print("[FAIL] NO REAL DATA FOUND")
    print("="*70)
    print("\nActionable Steps:")
    print("1. Download historical dataset from GitHub")
    print("   - Search: 'Stocktwits historical data github'")
    print("   - Save to: data/historical/{symbol}_messages.csv")
    print("2. Use Selenium for live scraping (more robust)")
    print("3. Contact Stocktwits for API access")
    print("4. Use alternative data sources (Reddit, Twitter)")
    
    return None

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'GME'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    result = collect_real_data(symbol, days)
    
    if result is not None and len(result) > 0:
        print(f"\n[OK] Real data collected: {len(result)} messages")
        print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        print(f"Sample: {result.iloc[0]['body'][:100]}...")
    else:
        print("\n[FAIL] Could not collect real data")

