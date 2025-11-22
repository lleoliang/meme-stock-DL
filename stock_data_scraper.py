import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker: str, days: int, end_date: str) -> pd.DataFrame:
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


def load_sp500_tickers(csv_path='sp500_companies.csv') -> list[str]:
    """
    Load S&P 500 ticker symbols from CSV
    
    Args:
        csv_path: Path to CSV file with 'Symbol' column
    
    Returns:
        List of ticker symbols
    """
    df = pd.read_csv(csv_path)
    tickers = df['Symbol'].tolist()
    print(f"Loaded {len(tickers)} S&P 500 tickers")
    return tickers


def save_to_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Save data to CSV file
    
    Args:
        data: DataFrame from get_stock_data
        filename: Like 'GME.csv'
    """
    data.to_csv(filename)
    print(f"Saved to {filename}")


def get_top_gainers(tickers: list, days: int, end_date: int, top_n: int = 100) -> list[str]:
    """
    Get top N gainers over a period
    
    Args:
        tickers: List of ticker symbols
        days: Lookback period (14 days)
        end_date: Reference date
        top_n: Number of top gainers to return (100)
    
    Returns:
        List of tuples: (ticker, return, data)
    """
    results = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(tickers)}")

        data = get_stock_data(ticker, days, end_date)
        if data is None or len(data) < 2:
            continue

        # Extract first-close and last-close
        first_close = float(data["Close"].iloc[0])
        last_close = float(data["Close"].iloc[-1])

        if first_close == 0:
            continue  # avoid division by zero

        pct_change = (last_close - first_close) / first_close

        results.append((ticker, pct_change, data))

    print(f"\nSuccessfully processed {len(results)} tickers")

    # Sort by percent change descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Slice top-N
    top_gainers = results[:top_n]

    if top_gainers:
        best = top_gainers[0]
        worst = top_gainers[-1]
        print(f"\nTop {top_n} gainers:")
        print(f"  Best: {best[0]} (+{best[1]*100:.2f}%)")
        print(f"  Worst in top {top_n}: {worst[0]} (+{worst[1]*100:.2f}%)")

    return top_gainers

if __name__ == "__main__":
    # Get GME data for 60 days before November 16, 2025
    ticker = 'GME'
    days = 14
    end_date = '2025-11-16'
    
    print(f"Fetching {ticker} for {days} days before {end_date}...")
    
    data = get_stock_data(ticker, days, end_date)
    
    # ===== Displaying the data =====
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
    # ===============================
    else:
        print("Failed to get data")

    universe = load_sp500_tickers(csv_path='sp500_companies.csv')
    top_100_tickers = get_top_gainers(universe, 14, "2025-11-01", 100)
    