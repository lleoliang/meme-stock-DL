"""
===============================================================
Data Extraction Pipeline for Meme-Stock Deep Learning Project
===============================================================

This script downloads historical OHLCV data for a universe of tickers 
(S&P 500), computes percent-change over a lookback window, selects 
the top-N gainers, and converts each ticker into a DataSample object.

A DataSample object can be thought of the tuple (M, p, y) as laid out in the 
mathematical framework. We then only need to add S to the tuple to form a complete
sample of (M, S, p, y)

Closely observe the DataSample class. Note how the dates are saved, so this should help
in extracting the data for S. 

Usage:
    Adjust these HYPERPARAMETERS/CONFIGURATION as desired. Ensure that when collecting data
    END_DATE is atleast a month apart from the previous end dates. 

Saved Output
------------
Each DataSample is exported as a compressed .npz file with fields:
    • ticker
    • dates
    • M
    • p
    • y
and exported into a folder labeled "exported_samples_{END_DATE}" for easy labeling and storage.

Example save locations:
    exported_samples_2025-11-01/AAPL.npz
    exported_samples_2025-11-01/MSFT.npz
    ...

How to Load Samples on Another Machine
--------------------------------------
Use this code for an individual DataSample:

    from stock_data_scraper import load_sample

    sample = load_sample("exported_samples_2025-11-01/AAPL.npz")
    print(sample)

This reconstructs the DataSample dataclass exactly.
"""

import yfinance as yf
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass 


# HYPERPARAMETERS/CONFIGURATION
END_DATE = "2025-11-01"
DAYS = 14
TOP_N = 100
SURGE_CUTOFF = 0.20


# Everything is type hinted and consistent with the general mathematical framework.
# Most of the framework and logic was created manually to debug more effectively.
# Specific decisions are justified accordingly with comments.


@dataclass
class DataSample:
    # Since dates are not stored well in NDArrays, we create a seperate array dates[]
    #   that is bijective to M, p, and y.
    # TODO: Extract -> add S -> rewrap in a new @dataclass
    ticker: str
    dates: list[str]  # Dates corresponding to each row
    M: NDArray[np.float64]  # OHLCV \in \R^{Tx5} where T \leq 14 (because market closes on weekends)
    p: NDArray[np.float64]  # p \in \R^{6}
    y: int  # 0 or 1

    def __post_init__(self):
        T = len(self.dates)

        assert self.M.shape[0] == T, "M's time dimension must match dates"
        assert self.p.ndim == 1 and self.p.shape[0] == 6, "p must be 6-dim window-level vector"
        assert self.y in (0, 1), "y must be 0 or 1"

    def __str__(self) -> str:
        T = len(self.dates)
        date_info = f"{self.dates[0]} → {self.dates[-1]}" if T > 1 else self.dates[0]

        # Build a small preview of the matrix M
        # (first 3 rows only so printing doesn't get out of hand)
        M_preview = np.array2string(self.M[:3], precision=3, suppress_small=True)
        if T > 3:
            M_preview += "\n    ..."

        p_str = np.array2string(self.p, precision=4, suppress_small=True)

        return (
            f"DataSample(\n"
            f"  ticker = '{self.ticker}',\n"
            f"  length = {T} days,\n"
            f"  dates  = {date_info},\n"
            f"  y      = {self.y},\n"
            f"  p (6,) = {p_str},\n"
            f"  M (first rows):\n"
            f"    {M_preview}\n"
            f")"
        )

def save_sample(sample: DataSample, folder: str) -> str:
    """
    Save one DataSample to a compressed .npz file.

    Fields saved:
        ticker (string)
        dates (string array)
        M (Tx5)
        p (6,)
        y (scalar)

    Args:
        sample: the DataSample object
        folder: the folder to save it in

    Returns:
        Full path to the saved file.
    """
    os.makedirs(folder, exist_ok=True)

    outpath = os.path.join(folder, f"{sample.ticker}.npz")
    
    np.savez_compressed(
        outpath,
        ticker=sample.ticker,
        dates=np.array(sample.dates),
        M=sample.M,
        p=sample.p,
        y=np.array(sample.y)
    )
    
    return outpath


def load_sample(path: str) -> DataSample:
    """
    Load a DataSample saved with save_sample().

    Args:
        path: the path to the individual DataSample

    Returns:
        A DataSample object
    """
    data = np.load(path, allow_pickle=True)

    return DataSample(
        ticker=str(data["ticker"]),
        dates=list(data["dates"]),
        M=data["M"],
        p=data["p"],
        y=int(data["y"])
    )


def get_stock_data(ticker: str, days: int, end_date: str) -> pd.DataFrame | None:
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
        auto_adjust=False,
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

    Returns:
        None
    """
    data.to_csv(filename)
    print(f"Saved to {filename}")


def extract_parabolic_features(data: pd.DataFrame) -> NDArray[np.float64]:
    """
    Extract 6 global parabolic breakout features for the entire window.

    Args: 
        data: OHLCV data matrix of (Tx5) size

    Returns:
        An NDArray of shape (6,)
    """

    close  = np.asarray(data["Close"],  float).reshape(-1)
    volume = np.asarray(data["Volume"], float).reshape(-1)
    high   = np.asarray(data["High"],   float).reshape(-1)
    low    = np.asarray(data["Low"],    float).reshape(-1)

    # Daily returns
    returns = (close[1:] - close[:-1]) / close[:-1]

    # Price acceleration
    accel = returns[1:] - returns[:-1] if len(returns) > 1 else np.array([0.0])

    # Volume ratios
    vol_ratio = volume[1:] / volume[:-1]

    # Range
    daily_range = high - low

    # ----- Global window summary (6 features) -----
    p = np.array([
        returns.mean(),
        returns.max(),
        accel.mean() if len(accel) > 0 else 0.0,
        vol_ratio.mean(),
        vol_ratio.max(),
        daily_range.mean()
    ], dtype=float)

    return p


def get_top_gainers(tickers: list[str], 
                    days: int, 
                    end_date: str,
                    top_n: int = 100) -> list[tuple[str, float, pd.DataFrame]]:
    """
    Get top N gainers over a period
    
    Args:
        tickers: List of ticker symbols
        days: Lookback period (14 days)
        end_date: Reference date
        top_n: Number of top gainers to return (100)
    
    Returns:
        List of tuples: (ticker, percent_change, data)
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

        percent_change = (last_close - first_close) / first_close

        results.append((ticker, percent_change, data))

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


def extract_sample(ticker: str, 
                   percent_change: float, 
                   data: pd.DataFrame) -> DataSample:
    """
    Extract a single training sample (M, p, y) for a given stock.

    Args:
        ticker (str)
        percent_change (float)
        data (pd.DataFrame)

    Returns:
        A DataSample (as we defined)
    """
    dates = [str(d.date()) for d in data.index]

    M = data[["Open", "High", "Low", "Close", "Volume"]].values.astype(float)

    p = extract_parabolic_features(data)

    y = 1 if percent_change > SURGE_CUTOFF else 0

    return DataSample(ticker, dates, M, p, y)


if __name__ == "__main__":
    # =============== BEING SINGLE TEST CASE ================

    # Get GME data for 60 days before November 16, 2024
    ticker = 'GME'
    days = 14
    end_date = '2024-11-16'
    
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
    
    # ================= END SINGLE TEST CASE ==================

    # ================= BEGIN S&P 500 CASE ===================

    universe = load_sp500_tickers(csv_path='sp500_companies.csv')
    top_n_tickers = get_top_gainers(tickers=universe, days=DAYS, end_date=END_DATE, top_n=TOP_N)
    
    samples = []

    for ticker, pct_change, data in top_n_tickers:
        sample = extract_sample(ticker, pct_change, data)
        samples.append(sample)
    
    for sample in samples:
        path = save_sample(sample, f"exported_samples_{END_DATE}")
        print(f"Saved {sample.ticker} -> {path}")

     # ================= END S&P 500 CASE ===================
    