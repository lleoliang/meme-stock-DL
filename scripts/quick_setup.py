"""
Quick setup script - sets environment and tests RapidAPI
Run this after setting your RapidAPI key
"""
import os
import sys

# Set your RapidAPI key here or use environment variable
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', '0c58d4f423mshbc1b1abec428bbap12c454jsn177519e55586')

if not RAPIDAPI_KEY:
    print("Set RAPIDAPI_KEY environment variable or edit this script")
    sys.exit(1)

# Set environment
os.environ['RAPIDAPI_KEY'] = RAPIDAPI_KEY
os.environ['STOCKTWITS_API_SOURCE'] = 'rapidapi'

print("="*70)
print("QUICK SETUP - Testing RapidAPI Integration")
print("="*70)
print(f"\nRapidAPI Key: {RAPIDAPI_KEY[:20]}...")
print(f"API Source: rapidapi\n")

# Test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_collector_enhanced import EnhancedStocktwitsCollector

try:
    collector = EnhancedStocktwitsCollector()
    print(f"Collector initialized: {type(collector).__name__}")
    print(f"API Source: {collector.api_source}")
    print(f"RapidAPI Key: {'Set' if collector.rapidapi_key else 'Not set'}\n")
    
    # Try to get data
    print("Testing data collection for GME...")
    messages = collector.get_stocktwits_messages('GME', days_back=7)
    
    print(f"\nResults:")
    print(f"  Messages collected: {len(messages)}")
    
    if len(messages) > 0:
        sample = messages.iloc[0]['body'] if len(messages) > 0 else ''
        is_real = 'Message about' not in sample
        
        if is_real:
            print(f"  Status: REAL STOCKTWITS DATA")
            print(f"  Sample message: {sample[:100]}...")
        else:
            print(f"  Status: Using synthetic data (RapidAPI endpoint not available)")
            print(f"  Note: RapidAPI Stocktwits API may not exist or endpoint changed")
    
    print("\n" + "="*70)
    print("Setup complete. Code is ready to use.")
    print("="*70)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

