"""
BASIC Stock Data Fetcher
Simple script to get stock data and save it
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, days, end_date):
    """
    Get stock data for ONE ticker
    
    Args:
        ticker: Stock symbol like 'GME'
        days: How many days back (e.g., 60)
        end_date: Final date in format 'YYYY-MM-DD' (e.g., '2025-07-01')
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    # Calculate start_date from end_date
    end = datetime.strptime(end_date, '%Y-%m-%d')
    start = end - timedelta(days=days)
    
    print(f"Fetching {ticker} from {start.date()} to {end.date()}")
    
    # Download with specific date range
    data = yf.download(
        ticker, 
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )
    
    # Check if we got data
    if data.empty:
        print(f"No data for {ticker}")
        return None
    
    return data


def save_to_csv(data, filename):
    """
    Save data to CSV file
    
    Args:
        data: DataFrame from get_stock_data
        filename: Like 'GME.csv'
    """
    data.to_csv(filename)
    print(f"Saved to {filename}")


# =======================
# EXAMPLE: Run this code
# =======================

if __name__ == "__main__":
    # Get GME data for 60 days before July 1, 2025
    ticker = 'GME'
    days = 7
    end_date = '2025-07-01'
    
    print(f"Fetching {ticker} for {days} days before {end_date}...")
    
    data = get_stock_data(ticker, days, end_date)
    
    if data is not None:
        # Show first few rows
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Show last few rows
        print("\nLast 5 rows:")
        print(data.tail())
        
        # Show what columns we have
        print("\nColumns:", list(data.columns))
        
        # Show date range
        print(f"\nActual date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Number of trading days: {len(data)}")
        
        # Save it
        save_to_csv(data, f'{ticker}_data.csv')
    else:
        print("Failed to get data")