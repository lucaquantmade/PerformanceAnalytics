import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import datetime
import requests
import time

# Path configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'PerformancesClean')

# Fixed FMP API key - replace with your actual key
FMP_API_KEY = "vCkfTebw75vtEmbGfEHWG30p3UJ3PSW2"  # Enter your API key here
use_api = True  # Set to False to skip benchmark data download

# Function to load data
def load_data(filename, display_name):
    possible_paths = [
        os.path.join(base_dir, 'data', 'PerformancesClean', filename),
        os.path.join(base_dir, 'data', 'Performances', filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Loading data from: {path}")
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                return df
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
    
    print(f"Could not find data for {display_name}")
    return None

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

# Selected quants
selected_quant_files = ["STEADY US 100performance.csv", "STEADY US Tech 100performance.csv"]
selected_quant_names = ["STEADY US 100", "STEADY US Tech 100"]

# Set up weights (evenly distributed)
weights = {name: 0.5 for name in selected_quant_names}  # 50% each

# Load and process data
returns_df = pd.DataFrame()
benchmark_returns_df = pd.DataFrame()

# Load data for each quant
for quant_file, display_name in zip(selected_quant_files, selected_quant_names):
    df = load_data(quant_file, display_name)
    if df is not None:
        returns_df[display_name] = df["Returns"]
        benchmark_returns_df[display_name] = df["Benchmark"].pct_change()

# Fill NaN values and check if data was loaded
returns_df.fillna(0, inplace=True)
benchmark_returns_df.fillna(0, inplace=True)

if returns_df.empty:
    print("No data was loaded. Please check file paths.")
    exit()

# Calculate weighted portfolio returns
quant_data = pd.DataFrame(index=returns_df.index)
quant_data["Returns"] = 0

for quant_name in selected_quant_names:
    quant_data["Returns"] += returns_df[quant_name] * weights[quant_name]

# Calculate strategy value
quant_data["StrategyValue"] = 100 * (1 + quant_data["Returns"]).cumprod()

# Calculate drawdown (as negative values)
quant_data["Cummax"] = quant_data["StrategyValue"].cummax()
quant_data["Drawdown"] = ((quant_data["StrategyValue"] - quant_data["Cummax"]) / quant_data["Cummax"]) * 100

# Download benchmark data - S&P 100 Equal Weight and NASDAQ 100 Equal Weight
print("Downloading benchmark data...")
start_date_download = "2014-01-01"  # Start a bit earlier for better reference
end_date_download = datetime.datetime.now().strftime("%Y-%m-%d")

benchmark_data = None

if use_api and FMP_API_KEY != "YOUR_API_KEY_HERE":
    try:
        # Define benchmark tickers (FMP uses different symbols than Yahoo)
        benchmark_tickers = {
            "NASDAQ100_EW": "QQQE",  # NASDAQ 100 Equal Weight ETF
            "SP100_EW": "RSP"        # S&P 500 Equal Weight ETF
        }
        
        benchmark_data = pd.DataFrame()
        
        # Download data for each benchmark with delay between requests
        for benchmark_name, ticker in benchmark_tickers.items():
            print(f"Downloading {benchmark_name} (ticker: {ticker}) data...")
            
            # Get historical data from FMP API
            benchmark_df = get_fmp_historical_data(ticker, start_date_download, end_date_download)
            
            if benchmark_df is not None and not benchmark_df.empty:
                # FMP returns 'adjClose' instead of 'Adj Close'
                benchmark_data[benchmark_name] = benchmark_df['adjClose']
                print(f"Successfully downloaded {benchmark_name} data")
            else:
                print(f"Failed to download {benchmark_name} data. Trying alternative ticker...")
                
                # Try alternative tickers if main one fails
                if benchmark_name == "NASDAQ100_EW" and ticker == "QQQE":
                    # Try another ETF as alternative
                    print("Trying alternative for NASDAQ 100 Equal Weight...")
                    alt_ticker = "QQQ"  # Regular NASDAQ 100 ETF
                    benchmark_df = get_fmp_historical_data(alt_ticker, start_date_download, end_date_download)
                    
                    if benchmark_df is not None and not benchmark_df.empty:
                        benchmark_data[benchmark_name] = benchmark_df['adjClose']
                        print(f"Using {alt_ticker} as alternative for {benchmark_name}")
                
                elif benchmark_name == "SP100_EW" and ticker == "RSP":
                    # Try another ETF as alternative
                    print("Trying alternative for S&P Equal Weight...")
                    alt_ticker = "SPY"  # Regular S&P 500 ETF
                    benchmark_df = get_fmp_historical_data(alt_ticker, start_date_download, end_date_download)
                    
                    if benchmark_df is not None and not benchmark_df.empty:
                        benchmark_data[benchmark_name] = benchmark_df['adjClose']
                        print(f"Using {alt_ticker} as alternative for {benchmark_name}")
            
            # Wait to avoid rate limiting
            time.sleep(1)
        
        # Check if we have both benchmark data series
        if "NASDAQ100_EW" in benchmark_data.columns and "SP100_EW" in benchmark_data.columns:
            # Calculate returns
            benchmark_data["NASDAQ100_EW_Returns"] = benchmark_data["NASDAQ100_EW"].pct_change()
            benchmark_data["SP100_EW_Returns"] = benchmark_data["SP100_EW"].pct_change()
            
            # Create a 50/50 blend of the two benchmarks
            benchmark_data["Blended_Returns"] = (benchmark_data["NASDAQ100_EW_Returns"] * 0.5) + (benchmark_data["SP100_EW_Returns"] * 0.5)
            
            # Calculate cumulative returns (starting at 100)
            benchmark_data["Blended_Value"] = 100 * (1 + benchmark_data["Blended_Returns"]).cumprod()
            
            # Replace NaNs with zeros
            benchmark_data.fillna(0, inplace=True)
            print("Successfully prepared benchmark data including 50/50 blend")
        else:
            if benchmark_data.empty:
                print("Could not download any benchmark data")
                benchmark_data = None
            else:
                print("Warning: Only partial benchmark data available")
        
    except Exception as e:
        print(f"Error in benchmark data preparation: {str(e)}")
        print("Continuing without benchmarks...")
        benchmark_data = None
else:
    if FMP_API_KEY == "YOUR_API_KEY_HERE":
        print("Please replace 'YOUR_API_KEY_HERE' with your actual FMP API key in the script.")
    else:
        print("API usage is disabled. Skipping benchmark data download.")
    print("To include benchmark comparisons, update the API key in the script.")
    benchmark_data = None

# Define the start date for 2015
calibration_date = pd.Timestamp('2015-01-01')
if calibration_date < quant_data.index.min():
    calibration_date = quant_data.index.min()
    print(f"Warning: Data doesn't go back to 2015. Using earliest available date: {calibration_date}")

# Filter data from calibration date onwards
filtered_data = quant_data.loc[calibration_date:].copy()

# Recalibrate to start at 100 in 2015
calibration_value = filtered_data["StrategyValue"].iloc[0]
filtered_data["StrategyValue"] = filtered_data["StrategyValue"] / calibration_value * 100

# Recalculate drawdown (as negative values)
filtered_data["Cummax"] = filtered_data["StrategyValue"].cummax()
filtered_data["Drawdown"] = ((filtered_data["StrategyValue"] - filtered_data["Cummax"]) / filtered_data["Cummax"]) * 100

# Prepare benchmark data if available
if benchmark_data is not None:
    # Filter to same date range and recalibrate
    benchmark_filtered = benchmark_data.loc[calibration_date:].copy()
    
    if not benchmark_filtered.empty:
        # Recalibrate benchmarks to 100
        nasdaq_calibration = benchmark_filtered["NASDAQ100_EW"].iloc[0]
        sp100_calibration = benchmark_filtered["SP100_EW"].iloc[0]
        blended_calibration = benchmark_filtered["Blended_Value"].iloc[0]
        
        benchmark_filtered["NASDAQ100_EW_Value"] = benchmark_filtered["NASDAQ100_EW"] / nasdaq_calibration * 100
        benchmark_filtered["SP100_EW_Value"] = benchmark_filtered["SP100_EW"] / sp100_calibration * 100
        benchmark_filtered["Blended_Value"] = benchmark_filtered["Blended_Value"] / blended_calibration * 100
        
        # Calculate benchmark drawdowns
        benchmark_filtered["NASDAQ100_EW_Cummax"] = benchmark_filtered["NASDAQ100_EW_Value"].cummax()
        benchmark_filtered["NASDAQ100_EW_Drawdown"] = ((benchmark_filtered["NASDAQ100_EW_Value"] - benchmark_filtered["NASDAQ100_EW_Cummax"]) / benchmark_filtered["NASDAQ100_EW_Cummax"]) * 100
        
        benchmark_filtered["SP100_EW_Cummax"] = benchmark_filtered["SP100_EW_Value"].cummax()
        benchmark_filtered["SP100_EW_Drawdown"] = ((benchmark_filtered["SP100_EW_Value"] - benchmark_filtered["SP100_EW_Cummax"]) / benchmark_filtered["SP100_EW_Cummax"]) * 100
        
        # Calculate blended benchmark drawdown (as negative values)
        benchmark_filtered["Blended_Cummax"] = benchmark_filtered["Blended_Value"].cummax()
        benchmark_filtered["Blended_Drawdown"] = ((benchmark_filtered["Blended_Value"] - benchmark_filtered["Blended_Cummax"]) / benchmark_filtered["Blended_Cummax"]) * 100
        
        # Calculate performance comparison metrics
        final_strategy_value = filtered_data["StrategyValue"].iloc[-1]
        final_benchmark_value = benchmark_filtered["Blended_Value"].iloc[-1]
        
        # Calculate total returns
        strategy_total_return = final_strategy_value - 100
        benchmark_total_return = final_benchmark_value - 100
        outperformance = strategy_total_return - benchmark_total_return
        
        # Calculate annualized returns
        days_count = (filtered_data.index[-1] - filtered_data.index[0]).days
        years = days_count / 365.25
        
        strategy_cagr = ((final_strategy_value / 100) ** (1 / years) - 1) * 100
        benchmark_cagr = ((final_benchmark_value / 100) ** (1 / years) - 1) * 100
        annual_outperformance = strategy_cagr - benchmark_cagr
        
        print(f"\nPerformance Comparison (since {calibration_date.strftime('%d.%m.%Y')}):")
        print(f"Total Period: {years:.1f} years")
        print(f"Quantmade Strategy Total Return: {strategy_total_return:.2f}%")
        print(f"Blended Benchmark Total Return: {benchmark_total_return:.2f}%")
        print(f"Total Outperformance: {outperformance:.2f}%")
        print(f"Quantmade Strategy CAGR: {strategy_cagr:.2f}%")
        print(f"Blended Benchmark CAGR: {benchmark_cagr:.2f}%")
        print(f"Annual Outperformance: {annual_outperformance:.2f}%")

# Create two subplots for performance and drawdown
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, dpi=300)
fig.subplots_adjust(hspace=0.1)  # Reduce space between subplots

# Format y-axis for performance chart
def format_y_axis(value, pos):
    return f'{value:.0f}'

ax1.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

# Plot strategy value on top chart
ax1.plot(filtered_data.index, filtered_data["StrategyValue"], 
         linewidth=2.5, color='#0066CC', label="Quantmade AI Strategie")

# Plot benchmarks if available
if benchmark_data is not None and not benchmark_filtered.empty:
    # Plot only the 50/50 blended benchmark
    ax1.plot(benchmark_filtered.index, benchmark_filtered["Blended_Value"], 
             linewidth=2, color='#FF5500', label="Equal Weight Benchmark (50/50 NASDAQ + S&P)")

# Plot drawdown on bottom chart
ax2.fill_between(filtered_data.index, filtered_data["Drawdown"], 0,
                 color='#0066CC', alpha=0.3, label="Quantmade AI Drawdown")

# Plot benchmark drawdowns if available
if benchmark_data is not None and not benchmark_filtered.empty:
    # Plot only the blended benchmark drawdown
    ax2.plot(benchmark_filtered.index, benchmark_filtered["Blended_Drawdown"], 
             linewidth=1.5, color='#FF5500', label="Equal Weight Benchmark Drawdown")

# Format x-axis to show years on both charts
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(years_fmt)
ax2.xaxis.set_major_locator(years)
ax2.xaxis.set_major_formatter(years_fmt)

# Hide x-axis labels on top chart
ax1.tick_params(axis='x', labelsize=10, labelbottom=False)
ax2.tick_params(axis='x', labelsize=10)

# Add grids
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(True, linestyle='--', alpha=0.7)

# Calculate min drawdown for proper y-axis scaling
min_drawdown = filtered_data["Drawdown"].min()
if benchmark_data is not None and not benchmark_filtered.empty:
    min_drawdown = min(min_drawdown, benchmark_filtered["Blended_Drawdown"].min())

# Set y-axis limits for drawdown chart (negative values, so bottom is min and top is 0)
min_limit = min(min_drawdown * 1.1, -30)  # Ensure at least -30% is visible
ax2.set_ylim(bottom=min_limit, top=5)  # Allow a small positive margin for better visualization

# Add labels and titles with performance information if available
if benchmark_data is not None and not benchmark_filtered.empty:
    title = f'Quantmade AI Strategie vs. Benchmark seit {calibration_date.strftime("%d.%m.%Y")}'
    title += f' | Outperformance: {outperformance:+.1f}% ({annual_outperformance:+.1f}% p.a.)'
    ax1.set_title(title, fontsize=16, fontweight='bold')
else:
    ax1.set_title(f'Strategie Performance seit {calibration_date.strftime("%d.%m.%Y")} (Startpunkt = 100)', fontsize=16, fontweight='bold')

ax1.set_ylabel('Wert', fontsize=12)
ax2.set_ylabel('Drawdown (%) - Negativ dargestellt', fontsize=12)
ax2.set_xlabel('Datum', fontsize=12)

# Set specific y-axis limits for the performance chart
max_strategy = filtered_data["StrategyValue"].max()
max_benchmark = 0
if benchmark_data is not None and not benchmark_filtered.empty:
    max_benchmark = benchmark_filtered["Blended_Value"].max()

y_max = max(max_strategy, max_benchmark) * 1.1  # Add 10% margin
ax1.set_ylim(top=y_max)

# Add legends
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=10)

# Add performance comparison annotation if benchmark data is available
if benchmark_data is not None and not benchmark_filtered.empty:
    # Calculate position for annotation - place in the middle right area
    date_position = filtered_data.index[int(len(filtered_data) * 0.65)]  # 65% from left
    y_position = y_max * 0.45  # 45% of the height - more central
    
    # Format the annotation text
    annotation_text = "Performance Details:\n"
    annotation_text += f"Quantmade: {strategy_total_return:+.1f}% ({strategy_cagr:+.1f}% p.a.)\n"
    annotation_text += f"Benchmark: {benchmark_total_return:+.1f}% ({benchmark_cagr:+.1f}% p.a.)"
    
    # Add the annotation with a nice background
    ax1.annotate(annotation_text, 
                xy=(date_position, y_position),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"),
                fontsize=10,
                fontweight="bold")
    
    # Generate output filename based on performance
    performance_indicator = "outperforming" if outperformance > 0 else "underperforming"
    output_file = f'quantmade_vs_benchmark_{performance_indicator}_{abs(annual_outperformance):.1f}pct_negative_drawdown.png'
else:
    output_file = 'strategie_performance_negative_drawdown.png'

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Chart saved as {output_file}")

# Show the name of the file for confirmation
print(f"High-resolution chart created: {os.path.abspath(output_file)}")

# Close the figure to free memory
plt.close() 