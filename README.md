<<<<<<< HEAD
=======
# Quantmade AI Strategy Dashboard
>>>>>>> 9899509 (Final version for website (hopefully))

## Requirements

The app requires the following Python packages:
- streamlit==1.31.0
- pandas==2.1.3
- numpy==1.26.3
- matplotlib==3.8.2
- plotly==5.18.0
- pillow==10.1.0
- kaleido==0.2.1 (for chart downloads)
<<<<<<< HEAD
=======
- requests==2.31.0 (for API calls)

You can install these with:
```
pip install -r requirements.txt
```

## Financial Modelling Prep API Integration

The `strategy_chart.py` script now uses Financial Modelling Prep API to download benchmark data. 
To use this feature:

1. Register for a free API key at [Financial Modelling Prep](https://financialmodelingprep.com/developer)
2. Open the `strategy_chart.py` file and replace `YOUR_API_KEY_HERE` with your actual API key:
   ```python
   # Fixed FMP API key - replace with your actual key
   FMP_API_KEY = "YOUR_API_KEY_HERE"  # Enter your API key here
   ```

The free tier includes 250-300 API calls per day, which is sufficient for generating the benchmark comparison charts.

## Running the Chart Generation Script

To generate the high-resolution performance chart with benchmark comparisons:

```bash
python strategy_chart.py
```

The output file `strategie_performance_mit_benchmarks.png` will be saved in the project directory.
>>>>>>> 9899509 (Final version for website (hopefully))

