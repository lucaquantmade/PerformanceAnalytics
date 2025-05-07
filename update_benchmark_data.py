import pandas as pd
import numpy as np
import os
import requests
import time
import datetime
from pathlib import Path

# FMP API key
FMP_API_KEY = "vCkfTebw75vtEmbGfEHWG30p3UJ3PSW2"  # Using the key from strategy_chart.py

# Path configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data", "PerformancesClean")

# Files to update
files_to_update = [
    "STEADY US 100performance.csv",
    "STEADY US Tech 100performance.csv"
]

# Function to download historical stock data from Financial Modelling Prep API
def get_fmp_historical_data(symbol, from_date, to_date):
    """
    Download historical stock price data from Financial Modelling Prep API.
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to download data for
    from_date : str
        Start date in YYYY-MM-DD format
    to_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with historical data or None if download fails
    """
    try:
        print(f"Downloading {symbol} data from FMP API...")
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {
            'from': from_date,
            'to': to_date,
            'apikey': FMP_API_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        
        if 'historical' not in data:
            print(f"Error: No historical data found for {symbol}")
            print(f"Response: {data}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['historical'])
        
        # Convert date string to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Sort by date (oldest first)
        df = df.sort_index()
        
        print(f"Successfully downloaded {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol} data: {str(e)}")
        return None

# Download benchmark data
start_date_download = "2004-01-01"  # Match the earliest date in the data
end_date_download = datetime.datetime.now().strftime("%Y-%m-%d")

print(f"Downloading benchmark data from {start_date_download} to {end_date_download}...")

# Download equal-weight ETF data
benchmark_data = pd.DataFrame()

# Download NASDAQ 100 Equal Weight ETF (QQQE)
nasdaq100_ew = get_fmp_historical_data("QQQE", start_date_download, end_date_download)
if nasdaq100_ew is not None and not nasdaq100_ew.empty:
    benchmark_data["NASDAQ100_EW"] = nasdaq100_ew["adjClose"]
    print(f"Successfully downloaded NASDAQ 100 Equal Weight ETF data")
else:
    # Try alternative
    nasdaq100 = get_fmp_historical_data("QQQ", start_date_download, end_date_download)
    if nasdaq100 is not None and not nasdaq100.empty:
        benchmark_data["NASDAQ100_EW"] = nasdaq100["adjClose"]
        print(f"Using QQQ as fallback for NASDAQ 100")

# Add delay to avoid rate limiting
time.sleep(1)

# Download S&P 500 Equal Weight ETF (RSP)
sp500_ew = get_fmp_historical_data("RSP", start_date_download, end_date_download)
if sp500_ew is not None and not sp500_ew.empty:
    benchmark_data["SP500_EW"] = sp500_ew["adjClose"]
    print(f"Successfully downloaded S&P 500 Equal Weight ETF data")
else:
    # Try alternative
    sp500 = get_fmp_historical_data("SPY", start_date_download, end_date_download)
    if sp500 is not None and not sp500.empty:
        benchmark_data["SP500_EW"] = sp500["adjClose"]
        print(f"Using SPY as fallback for S&P 500")

# Create a 50/50 blended benchmark
if "NASDAQ100_EW" in benchmark_data.columns and "SP500_EW" in benchmark_data.columns:
    # Calculate benchmark daily returns
    benchmark_data["NASDAQ100_EW_Return"] = benchmark_data["NASDAQ100_EW"].pct_change()
    benchmark_data["SP500_EW_Return"] = benchmark_data["SP500_EW"].pct_change()
    
    # Create 50/50 blend of return streams
    benchmark_data["Blended_Return"] = (
        benchmark_data["NASDAQ100_EW_Return"] * 0.5 + 
        benchmark_data["SP500_EW_Return"] * 0.5
    )
    
    # Calculate cumulative value
    benchmark_data["Blended_Value"] = 100 * (1 + benchmark_data["Blended_Return"]).cumprod()
    
    # Fill NaN values (first row)
    benchmark_data["Blended_Value"] = benchmark_data["Blended_Value"].fillna(100)
    
    print("Successfully created 50/50 blended benchmark")
else:
    print("Could not create blended benchmark - missing data")
    exit(1)

# Create backup of original files
for file_name in files_to_update:
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        backup_path = os.path.join(data_dir, f"{file_name}.bak")
        print(f"Creating backup of {file_name} to {backup_path}")
        Path(file_path).replace(backup_path)

# Load each file and update
for file_name in files_to_update:
    file_path = os.path.join(data_dir, file_name)
    backup_path = os.path.join(data_dir, f"{file_name}.bak")
    
    print(f"Updating {file_name} with new equal-weight benchmark data...")
    
    # Load the original data
    try:
        df = pd.read_csv(backup_path, parse_dates=["Date"], index_col="Date")
        print(f"Loaded {len(df)} rows from {backup_path}")
    except Exception as e:
        print(f"Error loading {backup_path}: {str(e)}")
        continue
    
    # Trim benchmark data to match the date range of the file
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Create the new benchmark with 100 as starting value
    blended_benchmark = benchmark_data.loc[start_date:end_date, "Blended_Value"].copy()
    
    # Calculate a scale factor to ensure the benchmark starts at 100
    scale_factor = 100 / blended_benchmark.iloc[0]
    blended_benchmark = blended_benchmark * scale_factor
    
    # Create a new dataframe with the updated benchmark
    df_new = df.copy()
    
    # Replace the benchmark column with the new equal-weight blended benchmark
    df_new["Benchmark"] = blended_benchmark
    
    # Calculate benchmark returns
    df_new["Benchmark_Returns"] = df_new["Benchmark"].pct_change()
    
    # Recalculate any other metrics if needed
    
    # Save the updated file
    print(f"Saving updated data to {file_path}")
    df_new.to_csv(file_path)

print("Benchmark data update complete!") 