# Quantmade AI Quant Funds Strategie Dashboard

A Streamlit dashboard visualizing quant investment strategies.

## Features

- Performance and drawdown visualization
- Key metrics calculation (CAGR, Sharpe ratio, Max Drawdown, etc.)
- Monthly returns heatmap
- Return triangle
- Rolling returns analysis
- Multilingual support (English/German)

## Deployment on Streamlit Cloud

1. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect to your GitHub repository
3. Set the main file path to `app.py`
4. Deploy the app

## Requirements

The app requires the following Python packages:
- streamlit==1.31.0
- pandas==2.1.3
- numpy==1.26.3
- matplotlib==3.8.2
- plotly==5.18.0
- pillow==10.1.0
- kaleido==0.2.1 (for chart downloads)

## Local Development

To run the app locally:

```bash
cd streamlit_app
streamlit run app.py
```

## Embedding in WordPress

To embed the Streamlit app in WordPress:

1. Deploy the app on Streamlit Cloud
2. Use an iframe in your WordPress site:

```html
<iframe
  src="https://your-app-url.streamlit.app/?embed=true"
  height="800"
  width="100%"
  style="border: none;"
></iframe>
```

## Data Structure

The app expects data files in the following structure:
- `/data/PerformancesClean/STEADY US 100performance.csv`
- `/data/PerformancesClean/STEADY US Tech 100performance.csv`

Each CSV file should have the following columns:
- Date (index)
- Returns
- Benchmark 