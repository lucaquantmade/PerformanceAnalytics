import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np
import os
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Quantmade AI Quant Funds Strategie Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for improved styling
st.markdown("""
<style>
    /* Main elements */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Hide Streamlit deployment and "..." indicators */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    .loader-text, .css-1l4firl {display: none;}
    
    /* Dashboard header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    /* Card container styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        box-shadow: none;
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-row {
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-weight: 600;
        font-size: 0.875rem;
        color: #5a5c69;
    }
    
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2b3674;
    }
    
    .metric-value.positive {
        color: #10b981;
    }
    
    .metric-value.negative {
        color: #ef4444;
    }
    
    .metric-container {
        padding: 16px;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: none;
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
    }
    
    .metric-section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    /* Date selector styling */
    .date-selector-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        box-shadow: none;
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Chart container styling */
    .chart-container {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 1rem;
        box-shadow: none;
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        background-color: #f3f4f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4e73df;
        font-weight: bold;
    }
    
    /* Language selector styling */
    .language-selector {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    /* Responsive adjustments for small screens */
    @media (max-width: 768px) {
        .dashboard-header {
            flex-direction: column;
            align-items: start;
        }
        
        .metric-container {
            padding: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Language dictionaries
TRANSLATIONS = {
    'en': {
        # General UI
        'page_title': 'Quant Strategy Analysis',
        'language': 'Language',
        'select_language': 'Select language',
        'dashboard_title': 'Quantmade AI Quant Funds Strategie Dashboard',
        'dashboard_subtitle': 'Monitor and analyze quant strategy performance',
        
        # Date selection
        'select_start_date': 'Select start date',
        'select_period': 'Select period',
        'time_period': 'Time Period',
        
        # Charts and Metrics
        'performance_drawdown': 'Performance and Drawdown',
        'strategy_value': 'Strategy Value (Starting value = 100)',
        'benchmark': 'Benchmark',
        'drawdown': 'Drawdown (%)',
        'value': 'Value',
        'date': 'Date',
        'download_performance': 'Download Performance Chart as PNG',
        
        # Metrics
        'metrics': 'Metrics',
        'performance_metrics': 'Performance Metrics',
        'risk_metrics': 'Risk Metrics',
        'relative_metrics': 'Relative Metrics',
        'period_performance': 'Period Performance',
        'cagr': 'CAGR',
        'total_return': 'Total Return',
        'volatility': 'Volatility (p.a.)',
        'max_drawdown': 'Max Drawdown',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'tracking_error': 'Tracking Error',
        'information_ratio': 'Information Ratio',
        'beta': 'Beta',
        'alpha': 'Alpha (p.a.)',
        'avg_recovery': 'Avg Recovery Time',
        'max_recovery': 'Max Recovery Time',
        'days': 'days',
        'ytd': 'YTD',
        'one_year_perf': '1Y Performance',
        'three_year_perf': '3Y Performance',
        'five_year_perf': '5Y Performance',
        'strategy': 'Strategy',
        'benchmark_label': 'Benchmark',
        'strategy_vs_benchmark': 'Strategy vs. Benchmark',
        'key_metrics': 'Key Metrics',
        
        # Monthly returns
        'monthly_returns': 'Monthly Returns since {}',
        'month': 'Month',
        'year': 'Year',
        'download_heatmap': 'Download Monthly Returns Heatmap as PNG',
        
        # Return triangle
        'return_triangle': 'Return Triangle',
        'return_triangle_title': 'Return Triangle - Annualized Returns Between Entry and Exit Years',
        'exit_year': 'Exit Year',
        'entry_year': 'Entry Year',
        'annual_return': 'Annual Return %',
        'download_triangle': 'Download Return Triangle as PNG',
        'triangle_caption': 'The return triangle shows the annualized return (in %) for different entry and exit times. The diagonal shows the return for each calendar year.',
        
        # Rolling returns
        'rolling_returns': 'Rolling Returns Analysis',
        'three_months': '3 Months',
        'one_year': '1 Year',
        'three_years': '3 Years',
        'comparison': 'Comparison',
        'not_enough_data': 'Not enough data for {}-Rolling-Returns. At least {} trading days required.',
        'rolling_returns_title': '{}-Rolling-Returns (Avg: {:.2f}%, σ: {:.2f}%)',
        'average': 'Average: {:.2f}%',
        'std_dev_plus': '+1 Std.dev.: {:.2f}%',
        'std_dev_minus': '-1 Std.dev.: {:.2f}%',
        'maximum': 'Maximum: {:.2f}%',
        'minimum': 'Minimum: {:.2f}%',
        'avg_return': 'Avg Return: {:.2f}%',
        'std_dev': 'Standard deviation: {:.2f}%',
        'min': 'Min: {:.2f}%',
        'max': 'Max: {:.2f}%',
        'download_rolling': 'Download {}-Rolling-Returns as PNG',
        'comparison_title': 'Comparison of Rolling Returns Across Different Periods (Monthly Averages)',
        'return': 'Return (%)',
        'statistics': 'Statistics',
        'period': 'Period',
        'download_comparison': 'Download Comparison Plot as PNG',
        'detailed_statistics': 'Detailed Statistics',
        'avg_return_pct': 'Avg Return (%)',
        'std_dev_pct': 'Std.dev. (%)',
        'min_pct': 'Min (%)',
        'max_pct': 'Max (%)',
        
        # Dashboard sections
        'overview': 'Overview',
        'detailed_analysis': 'Detailed Analysis',
        'performance_analysis': 'Performance Analysis',
        'risk_analysis': 'Risk Analysis',
        'return_analysis': 'Return Analysis'
    },
    'de': {
        # General UI
        'page_title': 'Lernen Sie unsere Quants kennen',
        'language': 'Sprache',
        'select_language': 'Sprache auswählen',
        'dashboard_title': 'Quantmade AI Quant Funds Strategie Dashboard',
        'dashboard_subtitle': 'Überwachen und analysieren Sie die Performance von Quant-Strategien',
        
        # Date selection
        'select_start_date': 'Startdatum auswählen',
        'select_period': 'Zeitraum auswählen',
        'time_period': 'Zeitraum',
        
        # Charts and Metrics
        'performance_drawdown': 'Performance und Drawdown',
        'strategy_value': 'Strategiewert (Startwert = 100)',
        'benchmark': 'Benchmark',
        'drawdown': 'Drawdown (%)',
        'value': 'Wert',
        'date': 'Datum',
        'download_performance': 'Performance-Grafik als PNG herunterladen',
        
        # Metrics
        'metrics': 'Kennzahlen',
        'performance_metrics': 'Performance-Metriken',
        'risk_metrics': 'Risikokennzahlen',
        'relative_metrics': 'Relative Kennzahlen',
        'period_performance': 'Zeitraum-Performance',
        'cagr': 'CAGR',
        'total_return': 'Total Return',
        'volatility': 'Volatilität (p.a.)',
        'max_drawdown': 'Max Drawdown',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'tracking_error': 'Tracking Error',
        'information_ratio': 'Information Ratio',
        'beta': 'Beta',
        'alpha': 'Alpha (p.a.)',
        'avg_recovery': 'Durchschn. Erholungszeit',
        'max_recovery': 'Max. Erholungszeit',
        'days': 'Tage',
        'ytd': 'YTD',
        'one_year_perf': '1J Performance',
        'three_year_perf': '3J Performance',
        'five_year_perf': '5J Performance',
        'strategy': 'Strategie',
        'benchmark_label': 'Benchmark',
        'strategy_vs_benchmark': 'Strategie vs. Benchmark',
        'key_metrics': 'Kennzahlen',
        
        # Monthly returns
        'monthly_returns': 'Monatliche Returns seit {}',
        'month': 'Monat',
        'year': 'Jahr',
        'download_heatmap': 'Monatsreturn-Heatmap als PNG herunterladen',
        
        # Return triangle
        'return_triangle': 'Renditedreieck',
        'return_triangle_title': 'Renditedreieck - Annualisierte Renditen zwischen Einstiegs- und Ausstiegsjahr',
        'exit_year': 'Ausstiegsjahr',
        'entry_year': 'Einstiegsjahr',
        'annual_return': 'Jährliche Rendite %',
        'download_triangle': 'Renditedreieck als PNG herunterladen',
        'triangle_caption': 'Das Renditedreieck zeigt die annualisierte Rendite (in %) für verschiedene Ein- und Ausstiegszeitpunkte. Die Diagonale zeigt die Rendite für jeweils ein Kalenderjahr.',
        
        # Rolling returns
        'rolling_returns': 'Rolling-Returns Analyse',
        'three_months': '3 Monate',
        'one_year': '1 Jahr', 
        'three_years': '3 Jahre',
        'comparison': 'Vergleich',
        'not_enough_data': 'Nicht genügend Daten für {}-Rolling-Returns. Mindestens {} Handelstage werden benötigt.',
        'rolling_returns_title': '{}-Rolling-Returns (Ø: {:.2f}%, σ: {:.2f}%)',
        'average': 'Durchschnitt: {:.2f}%',
        'std_dev_plus': '+1 Std.abw.: {:.2f}%',
        'std_dev_minus': '-1 Std.abw.: {:.2f}%',
        'maximum': 'Maximum: {:.2f}%',
        'minimum': 'Minimum: {:.2f}%',
        'avg_return': 'Ø Return: {:.2f}%',
        'std_dev': 'Standardabweichung: {:.2f}%',
        'min': 'Min: {:.2f}%',
        'max': 'Max: {:.2f}%',
        'download_rolling': '{}-Rolling-Returns als PNG herunterladen',
        'comparison_title': 'Vergleich der Rolling-Returns über verschiedene Zeiträume (Monatsdurchschnitte)',
        'return': 'Return (%)',
        'statistics': 'Statistiken',
        'period': 'Zeitraum',
        'download_comparison': 'Vergleichsplot als PNG herunterladen',
        'detailed_statistics': 'Detaillierte Statistiken',
        'avg_return_pct': 'Ø Return (%)',
        'std_dev_pct': 'Std.abw. (%)',
        'min_pct': 'Min (%)',
        'max_pct': 'Max (%)',
        
        # Dashboard sections
        'overview': 'Übersicht',
        'detailed_analysis': 'Detaillierte Analyse',
        'performance_analysis': 'Performance-Analyse',
        'risk_analysis': 'Risiko-Analyse',
        'return_analysis': 'Rendite-Analyse'
    }
}

# Function to create download link for plotly figures
def get_image_download_link(fig, filename, text, width=1200, height=800, scale=2):
    """
    Generates a download link for a plotly figure as a high-res PNG with transparent background.
    Checks for kaleido package and provides installation instructions if missing.
    
    Parameters:
    -----------
    fig : plotly.graph_objs.Figure
        The plotly figure to export
    filename : str
        Name of the file to download
    text : str
        Label for the download button
    width : int, optional
        Width of the exported image in pixels (default: 1200)
    height : int, optional
        Height of the exported image in pixels (default: 800)
    scale : int, optional
        Scale factor for resolution (default: 2)
    """
    try:
        # Set transparent background using rgba(0,0,0,0) and higher resolution
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Generate high-res PNG image
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        
        # Create download button
        return st.download_button(
            label=text,
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception as e:
        if "kaleido" in str(e).lower():
            st.warning("""
            💡 Um Grafiken als PNG herunterzuladen, muss das Paket 'kaleido' installiert werden.
            
            Installation über die Kommandozeile:
            ```
            pip install -U kaleido
            ```
            
            Nach der Installation starten Sie die App neu.
            """)
        else:
            st.error(f"Fehler beim Exportieren der Grafik: {str(e)}")
        return None

# Function to calculate recovery times from drawdowns
def calculate_recovery_times(data):
    """Calculate average and maximum recovery time from drawdowns"""
    # Identify peaks (where the strategy value equals its cumulative max)
    data['is_peak'] = (data['StrategyValue'] == data['Cummax'])
    
    # Identify recovery points
    recovery_times = []
    in_drawdown = False
    drawdown_start = None
    peak_value = None
    
    for i in range(len(data)):
        if data['is_peak'].iloc[i] and not in_drawdown:
            # At a new peak, not in drawdown
            peak_value = data['StrategyValue'].iloc[i]
        
        elif not data['is_peak'].iloc[i] and not in_drawdown:
            # Starting a drawdown
            in_drawdown = True
            drawdown_start = data.index[i]
        
        elif in_drawdown and data['StrategyValue'].iloc[i] >= peak_value:
            # Recovered from drawdown
            in_drawdown = False
            recovery_end = data.index[i]
            recovery_time = (recovery_end - drawdown_start).days
            
            # Only count significant drawdowns (e.g., > 5%)
            max_drawdown_in_period = data.loc[drawdown_start:recovery_end, 'drawdown'].min()
            if abs(max_drawdown_in_period) > 5:  # Only significant drawdowns > 5%
                recovery_times.append(recovery_time)
    
    if not recovery_times:
        return 0, 0  # No recovery times found
    
    avg_recovery = sum(recovery_times) / len(recovery_times)
    max_recovery = max(recovery_times)
    
    return avg_recovery, max_recovery

# Session state initialization for language
if 'language' not in st.session_state:
    st.session_state.language = 'de'  # Default to German

# Get translation function
def t(key):
    """Get translated text for the key in current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# Top language selector
col1, col2 = st.columns([6, 1])
with col2:
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("DE", use_container_width=True, 
                    disabled=st.session_state.language == 'de',
                    type="primary" if st.session_state.language == 'de' else "secondary"):
            st.session_state.language = 'de'
            st.rerun()
    
    with lang_col2:
        if st.button("EN", use_container_width=True,
                    disabled=st.session_state.language == 'en',
                    type="primary" if st.session_state.language == 'en' else "secondary"):
            st.session_state.language = 'en'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Pre-selected quants - file names may have duplicated "performance.csv"
selected_quant_files = ["STEADY US 100performance.csvperformance.csv", "STEADY US Tech 100performance.csvperformance.csv"]
selected_quant_names = ["STEADY US 100", "STEADY US Tech 100"]
risk_free_rate = 2.0 / 100  # Default 2.0%

# Set up weights (evenly distributed)
weights = {}
for quant_name in selected_quant_names:
    weights[quant_name] = 0.5  # 50% each

# Process data for selected quants
returns_df = pd.DataFrame()
benchmark_returns_df = pd.DataFrame()

# Use path that works both locally and in cloud
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data - use os.path.join for cross-platform compatibility
for quant, display_name in zip(selected_quant_files, selected_quant_names):
    data_file = os.path.join(base_dir, 'data', 'PerformancesClean', quant)
    try:
        df = pd.read_csv(data_file, parse_dates=["Date"], index_col="Date")
        returns_df[display_name] = df["Returns"]
        benchmark_returns_df[display_name] = df["Benchmark"].pct_change()
    except FileNotFoundError:
        st.error(f"Data file not found: {data_file}")
        st.stop()

# Fill NaN values
returns_df.fillna(0, inplace=True)
benchmark_returns_df.fillna(0, inplace=True)

# Calculate weighted portfolio returns
quant_data = pd.DataFrame(index=returns_df.index)
quant_data["Returns"] = 0
quant_data["Benchmark"] = 0
quant_data["Benchmark_Returns"] = 0

for quant_name in selected_quant_names:
    quant_data["Returns"] += returns_df[quant_name] * weights[quant_name]
    quant_data["Benchmark_Returns"] += benchmark_returns_df[quant_name] * weights[quant_name]

# Calculate additional metrics
quant_data["StrategyValue"] = 100 * (1 + quant_data["Returns"]).cumprod()
quant_data["Benchmark"] = 100 * (1 + quant_data["Benchmark_Returns"]).cumprod()
quant_data["drawdown"] = -((quant_data["StrategyValue"].cummax() - quant_data["StrategyValue"])/quant_data["StrategyValue"].cummax())*100
quant_data["monthly_returns"] = (1 + quant_data["Returns"]).resample("M").prod() - 1

# Dashboard header with strategy name
strategy_display_name = " + ".join(selected_quant_names)
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <h1>{t('dashboard_title')}: {strategy_display_name}</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# Date selection in an elegant container
st.markdown(f"""<div class="date-selector-container"><h4>📅 {t('time_period')}</h4>""", unsafe_allow_html=True)

# Date selector variables
min_date = quant_data.index.min().date()
max_date = quant_data.index.max().date()

# All available dates as list for selector
available_dates = quant_data.index.unique()
date_options = [min_date] + sorted(list(set(d.date() for d in available_dates if pd.notna(d))))

# Columns for date selection
date_cols = st.columns([2, 3])

# Time period buttons in first column
with date_cols[0]:
    period_options = ["YTD", t('one_year'), "3 " + t('three_years').split()[-1], "5 " + t('three_years').split()[-1], "MAX"]
    selected_period = st.radio(t('select_period'), options=period_options, horizontal=True, index=4)  # Default to MAX

# Date selection with slider
with date_cols[1]:
    # Selector for start date with actual date values
    selected_start_date = st.select_slider(
        t('select_start_date'),
        options=date_options,
        value=min_date,
        format_func=lambda x: x.strftime('%d.%m.%Y')
    )

# Apply period if one was selected
if selected_period == "YTD":
    current_year = datetime.datetime.now().year
    selected_start_date = datetime.date(current_year, 1, 1)
elif selected_period == t('one_year'):
    selected_start_date = max_date - datetime.timedelta(days=365)
elif selected_period == "3 " + t('three_years').split()[-1]:
    selected_start_date = max_date - datetime.timedelta(days=365*3)
elif selected_period == "5 " + t('three_years').split()[-1]:
    selected_start_date = max_date - datetime.timedelta(days=365*5)
elif selected_period == "MAX":
    selected_start_date = min_date

# Find closest date in available dates
start_date = selected_start_date
end_date = max_date

# Close the date selector container
st.markdown("</div>", unsafe_allow_html=True)

# Filter data by selected time period
filtered_data = quant_data.loc[start_date:end_date].copy()

# Recalculate values if start_date is not the first date
if start_date != min_date:
    # Scale data to 100, calculate drawdown after scaling
    first_value = filtered_data["StrategyValue"].iloc[0]
    first_benchmark = filtered_data["Benchmark"].iloc[0]
    
    filtered_data["StrategyValue"] = filtered_data["StrategyValue"] / first_value * 100
    filtered_data["Benchmark"] = filtered_data["Benchmark"] / first_benchmark * 100
    filtered_data["Cummax"] = filtered_data["StrategyValue"].cummax()
    filtered_data["drawdown"] = -((filtered_data["Cummax"] - filtered_data["StrategyValue"])/filtered_data["Cummax"])*100
    filtered_data["Monat"] = filtered_data.index.month
    filtered_data["monthly_returns"] = (1 + filtered_data["Returns"]).resample("M").prod() - 1
    filtered_data.fillna(0, inplace=True)

# Calculate metrics
max_drawdown = abs(filtered_data["drawdown"].min())
total_return = ((filtered_data["StrategyValue"].iloc[-1]/100) - 1) 
annual_volatility = (filtered_data["Returns"].std() * np.sqrt(252))
cagr = ((1 + total_return)**(252/len(filtered_data)) - 1)

# Sharpe ratio with adjustable risk-free rate
sharpe = (cagr - risk_free_rate)/annual_volatility if annual_volatility != 0 else 0

downside_returns = filtered_data["Returns"][filtered_data["Returns"] < 0]
downside_volatility = (downside_returns.std() * np.sqrt(252)) if not downside_returns.empty else 0.001
sortino = (cagr - risk_free_rate)/downside_volatility if downside_volatility != 0 else 0

# Tracking error - annualized standard deviation of return differences
return_difference = filtered_data["Returns"] - filtered_data["Benchmark_Returns"] 
tracking_error = return_difference.std() * np.sqrt(252) if return_difference.std() != 0 else 0.001

# Information ratio
information_ratio = (cagr - filtered_data["Benchmark_Returns"].mean() * 252) / tracking_error if tracking_error != 0 else 0

# Recovery time calculation
if "Cummax" not in filtered_data.columns:
    filtered_data["Cummax"] = filtered_data["StrategyValue"].cummax()
avg_recovery, max_recovery = calculate_recovery_times(filtered_data)

# beta and alpha with monthly returns
monthly_returns = filtered_data["Returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
monthly_returns.fillna(0, inplace=True)
benchmark_monthly_returns = filtered_data["Benchmark_Returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
benchmark_monthly_returns.fillna(0, inplace=True)

# Calculate beta
beta = monthly_returns.cov(benchmark_monthly_returns)/benchmark_monthly_returns.var() if benchmark_monthly_returns.var() != 0 else 0

# Calculate alpha and annualize it (convert to percentage)
alpha_monthly = monthly_returns.mean() - beta * benchmark_monthly_returns.mean()
alpha_annualized_pct = alpha_monthly * 12 * 100  # Annualize and convert to percentage

# Calculate period performance metrics
# Get today's date and current year start for YTD calculation
today = pd.Timestamp.now()
ytd_start = pd.Timestamp(today.year, 1, 1)
one_year_ago = today - pd.DateOffset(years=1)
three_years_ago = today - pd.DateOffset(years=3)
five_years_ago = today - pd.DateOffset(years=5)

# Find closest dates in data
all_dates = quant_data.index
ytd_start_date = all_dates[all_dates >= ytd_start].min() if any(all_dates >= ytd_start) else all_dates.max()
one_year_date = all_dates[all_dates >= one_year_ago].min() if any(all_dates >= one_year_ago) else all_dates.min()
three_year_date = all_dates[all_dates >= three_years_ago].min() if any(all_dates >= three_years_ago) else all_dates.min()
five_year_date = all_dates[all_dates >= five_years_ago].min() if any(all_dates >= five_years_ago) else all_dates.min()

# Calculate performance values
last_date = quant_data.index.max()

# Strategy performance
ytd_perf = (quant_data.loc[last_date, 'StrategyValue'] / quant_data.loc[ytd_start_date, 'StrategyValue'] - 1) * 100 if ytd_start_date != last_date else 0
one_year_perf = (quant_data.loc[last_date, 'StrategyValue'] / quant_data.loc[one_year_date, 'StrategyValue'] - 1) * 100 if one_year_date != last_date else 0
three_year_perf = (quant_data.loc[last_date, 'StrategyValue'] / quant_data.loc[three_year_date, 'StrategyValue'] - 1) * 100 if three_year_date != last_date else 0
five_year_perf = (quant_data.loc[last_date, 'StrategyValue'] / quant_data.loc[five_year_date, 'StrategyValue'] - 1) * 100 if five_year_date != last_date else 0

# Benchmark performance
ytd_bench_perf = (quant_data.loc[last_date, 'Benchmark'] / quant_data.loc[ytd_start_date, 'Benchmark'] - 1) * 100 if ytd_start_date != last_date else 0
one_year_bench_perf = (quant_data.loc[last_date, 'Benchmark'] / quant_data.loc[one_year_date, 'Benchmark'] - 1) * 100 if one_year_date != last_date else 0
three_year_bench_perf = (quant_data.loc[last_date, 'Benchmark'] / quant_data.loc[three_year_date, 'Benchmark'] - 1) * 100 if three_year_date != last_date else 0
five_year_bench_perf = (quant_data.loc[last_date, 'Benchmark'] / quant_data.loc[five_year_date, 'Benchmark'] - 1) * 100 if five_year_date != last_date else 0

# Create main dashboard tabs
overview_tab, returns_tab, rolling_tab, details_tab = st.tabs([
    "📊 " + t('overview'),
    "💹 " + t('return_analysis'),
    "📈 " + t('rolling_returns'),
    "📋 " + t('detailed_analysis')
])

# OVERVIEW TAB - Main dashboard metrics and performance chart
with overview_tab:
    # Key Performance Indicators
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📊 Key Performance Indicators</h3>
    """, unsafe_allow_html=True)
    
    # Layout: key metrics in flexible grid
    col1, col2, col3, col4 = st.columns(4)
    
    # Format metrics with appropriate colors
    def format_metric(value, is_percentage=True, reverse_colors=False):
        formatted = f"{value:.2f}{'%' if is_percentage else ''}"
        if value > 0:
            css_class = "positive" if not reverse_colors else "negative"
        elif value < 0:
            css_class = "negative" if not reverse_colors else "positive"
        else:
            css_class = ""
        return f'<span class="metric-value {css_class}">{formatted}</span>'
    
    # First row of metrics - main performance indicators
    with col1:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('cagr')}</div>
            {format_metric(cagr*100)}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('total_return')}</div>
            {format_metric(total_return*100)}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('alpha')}</div>
            {format_metric(alpha_annualized_pct)}
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('beta')}</div>
            {format_metric(beta, is_percentage=False)}
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of metrics - risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('volatility')}</div>
            {format_metric(annual_volatility*100, reverse_colors=True)}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('max_drawdown')}</div>
            {format_metric(max_drawdown, reverse_colors=True)}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('sortino_ratio')}</div>
            {format_metric(sortino, is_percentage=False)}
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('tracking_error')}</div>
            {format_metric(tracking_error*100, reverse_colors=True)}
        </div>
        """, unsafe_allow_html=True)
    
    # Section for period performance comparison
    st.markdown(f"""<h4>{t('strategy_vs_benchmark')}</h4>""", unsafe_allow_html=True)
    
    # Create period performance table with 4 columns
    period_cols = st.columns(4)
    
    # YTD Performance
    with period_cols[0]:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('ytd')}</div>
            <div>
                {t('strategy')}: {format_metric(ytd_perf)}<br>
                {t('benchmark_label')}: {format_metric(ytd_bench_perf)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 1Y Performance
    with period_cols[1]:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('one_year_perf')}</div>
            <div>
                {t('strategy')}: {format_metric(one_year_perf)}<br>
                {t('benchmark_label')}: {format_metric(one_year_bench_perf)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 3Y Performance
    with period_cols[2]:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('three_year_perf')}</div>
            <div>
                {t('strategy')}: {format_metric(three_year_perf)}<br>
                {t('benchmark_label')}: {format_metric(three_year_bench_perf)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 5Y Performance
    with period_cols[3]:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">{t('five_year_perf')}</div>
            <div>
                {t('strategy')}: {format_metric(five_year_perf)}<br>
                {t('benchmark_label')}: {format_metric(five_year_bench_perf)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close the metrics container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance and Drawdown visualization
    st.markdown("""<div class="chart-container">
        <h3 class="metric-section-header">📈 """ + t('performance_drawdown') + """</h3>
    """, unsafe_allow_html=True)

    # Create subplot with two charts
    fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(t('strategy_value'), None),
                    row_heights=[0.67, 0.33])  # 2/3 for strategy value, 1/3 for drawdown

    # Strategy value plot with area filling
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index, 
            y=filtered_data['StrategyValue'],
            mode='lines',
            name=t('strategy_value'),
            line=dict(color='rgba(65, 105, 225, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.1)'
        ),
        row=1, col=1
    )
    
    # Benchmark plot with area filling
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['Benchmark'],
            mode='lines',
            name=t('benchmark'),
            line=dict(color='rgba(128, 128, 128, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ),
        row=1, col=1
    )

    # Drawdown plot with area filling
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['drawdown'],
            mode='lines',
            name=t('drawdown'),
            line=dict(color='rgba(128, 0, 128, 0.8)', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        ),
        row=2, col=1
    )

    # Adjust layout
    fig.update_layout(
        height=600,  # Reset to original height
        showlegend=True,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),  # Adjusted margins
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(size=12)  # Increased legend text size
        ),
        template="plotly_white"  # Use a clean white template
    )

    # Y-axis labels
    fig.update_yaxes(title_text=t('value'), row=1, col=1, tickfont=dict(size=12))  # Increased tick font size
    fig.update_yaxes(title_text=t('drawdown'), row=2, col=1, tickfont=dict(size=12))  # Increased tick font size
    
    # X-axis between plots
    fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide X-axis in upper plot 
    fig.update_xaxes(
        title_text=t('date'), 
        row=2, 
        col=1, 
        title_standoff=0, 
        tickfont=dict(size=12)  # Increased tick font size
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button for performance chart
    strategy_name = "_".join(selected_quant_files) if len(selected_quant_files) > 1 else selected_quant_files[0]
    strategy_display_name_for_file = "_".join(selected_quant_names)
    period_text = selected_period if selected_period else f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    performance_filename = f"performance_drawdown_{strategy_display_name_for_file}_{period_text}.png"
    get_image_download_link(fig, performance_filename, "📥 " + t('download_performance'))
    
    # Close chart container
    st.markdown("</div>", unsafe_allow_html=True)

with returns_tab:
    # RETURNS ANALYSIS TAB - Monthly returns and return triangle
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📅 """ + t('monthly_returns').format(start_date.strftime("%d.%m.%Y")) + """</h3>
    """, unsafe_allow_html=True)
    
    # Set start date for heatmap to max(start_date, 2020-01-01)
    heatmap_start_date = max(pd.Timestamp(start_date), pd.Timestamp('2020-01-01'))
    heatmap_data = filtered_data.loc[heatmap_start_date:end_date].copy()
    
    # Calculate monthly returns (in percent)
    # Using point-to-point return calculation for consistency
    monthly_returns_df = pd.DataFrame()
    
    # Get data resampled to month-end
    monthly_prices = heatmap_data.resample('M')['StrategyValue'].last()
    
    # Calculate month-to-month returns properly
    for year in sorted(heatmap_data.index.year.unique()):
        year_data = monthly_prices[monthly_prices.index.year == year]
        if not year_data.empty:
            # For each month, calculate return based on previous month's value
            for i, (date, value) in enumerate(year_data.items()):
                if i == 0 and year > heatmap_data.index.year.min():
                    # For first month of non-first year, get last month of previous year
                    prev_year_last_month = monthly_prices[monthly_prices.index.year == year-1].iloc[-1] if not monthly_prices[monthly_prices.index.year == year-1].empty else None
                    if prev_year_last_month is not None:
                        monthly_returns_df.loc[date, 'Returns'] = ((value / prev_year_last_month) - 1) * 100
                    else:
                        # If no previous data, use first value in this year's data
                        first_day_value = heatmap_data[heatmap_data.index.year == year].iloc[0]['StrategyValue']
                        monthly_returns_df.loc[date, 'Returns'] = ((value / first_day_value) - 1) * 100
                elif i > 0:
                    # For subsequent months, calculate based on previous month
                    prev_month_value = year_data.iloc[i-1]
                    monthly_returns_df.loc[date, 'Returns'] = ((value / prev_month_value) - 1) * 100
                else:
                    # For very first month in dataset, use the first day's value
                    first_day_value = heatmap_data[heatmap_data.index.year == year].iloc[0]['StrategyValue']
                    monthly_returns_df.loc[date, 'Returns'] = ((value / first_day_value) - 1) * 100
    
    # Calculate YTD and yearly returns
    yearly_returns = {}
    for year in sorted(heatmap_data.index.year.unique()):
        # Get first and last value for the year
        year_data = heatmap_data[heatmap_data.index.year == year]
        if not year_data.empty:
            first_value = year_data.iloc[0]['StrategyValue']
            last_value = year_data.iloc[-1]['StrategyValue']
            yearly_returns[year] = ((last_value / first_value) - 1) * 100
    
    # Create the pivot table from the calculated monthly returns
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='Returns')
    
    # Add yearly returns column
    for year in pivot_table.index:
        if year in yearly_returns:
            pivot_table.loc[year, 'yearly'] = yearly_returns[year]
    
    # Add empty column for visual separation
    pivot_table.insert(len(pivot_table.columns) - 1, ' ', np.nan)

    # Update column names (including month names)
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mär' if st.session_state.language == 'de' else 'Mar', 
        4: 'Apr', 5: 'Mai' if st.session_state.language == 'de' else 'May', 
        6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 
        10: 'Okt' if st.session_state.language == 'de' else 'Oct', 
        11: 'Nov', 12: 'Dez' if st.session_state.language == 'de' else 'Dec',
        'yearly': 'Yearly' if st.session_state.language == 'en' else 'Jährlich'
    }
    
    # Rename considering the new separator column
    pivot_table.columns = [
        month_names.get(col, col) if isinstance(col, (int, str)) else col
        for col in pivot_table.columns
    ]
    
    # Replace NaN values
    pivot_table.fillna(np.nan, inplace=True)
    
    # Create text values for better readability
    text_values = []
    for i in range(len(pivot_table.index)):
        row_texts = []
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            if pd.notna(value):
                # Add a plus sign for positive values for easier reading
                sign = "+" if value > 0 else ""
                row_texts.append(f"{sign}{value:.1f}%")  # Round to 1 decimal place
            else:
                row_texts.append("")
        text_values.append(row_texts)
        
    # Create a custom colorscale with darker colors for better text contrast
    # Red-Yellow-Green with stronger colors
    custom_colorscale = [
        [0, 'rgb(165,0,38)'],      # Dark red
        [0.3, 'rgb(215,48,39)'],   # Red
        [0.45, 'rgb(244,109,67)'], # Light red-orange
        [0.5, 'rgb(253,174,97)'],  # Light orange
        [0.55, 'rgb(254,224,144)'],# Very light yellow
        [0.65, 'rgb(217,239,139)'],# Light green-yellow
        [0.8, 'rgb(145,207,96)'],  # Light green
        [1, 'rgb(26,152,80)']      # Dark green
    ]
        
    # Create heatmap with Plotly
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale=custom_colorscale,  # Custom colorscale for better contrast
        zmin=-10,                      # Lower limit for color scale
        zmid=-1,                       # Shift of color transition - Yellow at 1% instead of at 0%
        zmax=8,                        # Upper limit for color scale
        text=text_values,              # Show rounded values
        hoverinfo='text',
        texttemplate='%{text}',
        textfont={"size": 13, "color": "black", "family": "Arial Black"},  # Bold, black text
        colorbar=dict(title=t('return'))
    ))
    
    # Adjust layout
    fig_heatmap.update_layout(
        title=t('monthly_returns').format(start_date.strftime("%d.%m.%Y")),
        xaxis_title=t('month'),
        yaxis_title=t('year'),
        height=500,  # Increased height
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=50),  # Adjusted margins
        yaxis=dict(autorange="reversed"),  # Newest year at top
        xaxis=dict(side="top"),  # x-axis displayed at top
        template="plotly_white"  # Use a clean white template
    )
    
    # Display heatmap
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Add download button for heatmap with wider width (1.3x)
    heatmap_filename = f"monthly_returns_{strategy_display_name_for_file}_{period_text}.png"
    get_image_download_link(
        fig_heatmap, 
        heatmap_filename, 
        "📥 " + t('download_heatmap'), 
        width=int(1200 * 1.3),  # 1.3 times wider
        height=800
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Return triangle creation
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📐 """ + t('return_triangle') + """</h3>
    """, unsafe_allow_html=True)
    
    # Calculate annual values (for all data, not just filtered)
    yearly_data = filtered_data.copy()
    
    # Create a DataFrame to store the first and last values of each year
    year_points = pd.DataFrame(columns=['year', 'first_date', 'first_value', 'last_date', 'last_value'])
    
    # Extract first and last value for each year
    for year in sorted(yearly_data.index.year.unique()):
        year_slice = yearly_data[yearly_data.index.year == year]
        if not year_slice.empty:
            year_points = year_points._append({
                'year': year,
                'first_date': year_slice.index.min(),
                'first_value': year_slice.loc[year_slice.index.min(), 'StrategyValue'],
                'last_date': year_slice.index.max(),
                'last_value': year_slice.loc[year_slice.index.max(), 'StrategyValue']
            }, ignore_index=True)
    
    # List of years
    years = sorted(year_points['year'].unique())
    
    # Create empty DataFrame for return triangle
    rendite_dreieck = pd.DataFrame(index=years, columns=years)
    
    # For each possible entry point (year)
    for i, start_year in enumerate(years):
        start_row = year_points[year_points['year'] == start_year].iloc[0]
        
        # For each possible exit point (year)
        for j, end_year in enumerate(years):
            if end_year >= start_year:
                end_row = year_points[year_points['year'] == end_year].iloc[0]
                
                # Calculate point-to-point return
                start_value = start_row['first_value']
                end_value = end_row['last_value']
                
                # Calculate total return for the period
                total_return = (end_value / start_value) - 1
                
                # For multi-year periods, annualize the return
                if end_year > start_year:
                    # Calculate the exact time difference in years (including partial years)
                    time_diff = (end_row['last_date'] - start_row['first_date']).days / 365.25
                    # Annualize the return
                    annualized_return = ((1 + total_return) ** (1 / time_diff)) - 1
                    rendite_dreieck.loc[start_year, end_year] = annualized_return * 100
                else:
                    # For single-year returns, just use the total return
                    rendite_dreieck.loc[start_year, end_year] = total_return * 100
    
    # Replace NaN values with empty strings for better display
    rendite_dreieck_display = rendite_dreieck.copy()
    rendite_dreieck_display = rendite_dreieck_display.fillna(np.nan)
    
    # Fill NaN values for heatmap
    rendite_dreieck_filled = rendite_dreieck.fillna(np.nan)
    
    # Create formatted values for text with improved readability
    text_values = []
    for i in range(len(rendite_dreieck.index)):
        row_texts = []
        for j in range(len(rendite_dreieck.columns)):
            value = rendite_dreieck.iloc[i, j]
            if pd.notna(value):
                # Add a plus sign for positive values for easier reading
                sign = "+" if value > 0 else ""
                row_texts.append(f"{sign}{value:.1f}%")  # Round to 1 decimal place
            else:
                row_texts.append("")
        text_values.append(row_texts)
    
    # Create a custom colorscale with darker colors for better text contrast
    # Red-Yellow-Green with stronger colors for triangle values
    custom_colorscale = [
        [0, 'rgb(165,0,38)'],      # Dark red
        [0.3, 'rgb(215,48,39)'],   # Red
        [0.45, 'rgb(244,109,67)'], # Light red-orange
        [0.5, 'rgb(253,174,97)'],  # Light orange
        [0.55, 'rgb(254,224,144)'],# Very light yellow
        [0.65, 'rgb(217,239,139)'],# Light green-yellow
        [0.8, 'rgb(145,207,96)'],  # Light green
        [1, 'rgb(26,152,80)']      # Dark green
    ]
    
    # Create heatmap for return triangle
    fig_dreieck = go.Figure(data=go.Heatmap(
        z=rendite_dreieck_filled.values,
        x=rendite_dreieck.columns,
        y=rendite_dreieck.index,
        colorscale=custom_colorscale,  # Custom colorscale for better contrast
        zmin=-15,
        zmid=0,
        zmax=15,
        text=text_values,  # Use pre-formatted text values
        hoverinfo='text',
        texttemplate='%{text}',  # Use text directly without further formatting
        textfont={"size": 13, "color": "black", "family": "Arial Black"},  # Bold, black text
        colorbar=dict(title=t('annual_return')),
        showscale=True
    ))
    
    # Adjust layout
    fig_dreieck.update_layout(
        title=t('return_triangle_title'),
        xaxis_title=t('exit_year'),
        yaxis_title=t('entry_year'),
        height=700,  # Increased height
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=80),  # Adjusted margins
        xaxis=dict(
            dtick=1,  # Show every year
            tickangle=45,  # Angled labels for better readability
            tickfont=dict(size=12)  # Increased tick font size
        ),
        yaxis=dict(
            dtick=1,  # Show every year
            autorange="reversed",  # Oldest years at top
            tickfont=dict(size=12)  # Increased tick font size
        ),
        template="plotly_white"  # Use a clean white template
    )
    
    # Display return triangle
    st.plotly_chart(fig_dreieck, use_container_width=True)
    
    # Add download button for return triangle
    dreieck_filename = f"renditedreieck_{strategy_display_name_for_file}_{period_text}.png"
    get_image_download_link(
        fig_dreieck, 
        dreieck_filename, 
        "📥 " + t('download_triangle'),
        width=int(1200 * 1.3),  # 1.3 times wider
        height=800
    )
    
    # Interpretation hint
    st.caption(t('triangle_caption'))
    st.markdown("</div>", unsafe_allow_html=True)

with rolling_tab:
    # Rolling returns calculation and visualization
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📈 """ + t('rolling_returns') + """</h3>
    """, unsafe_allow_html=True)
    
    # Trading days for different periods
    rolling_window_3m = 63     # 3 months (21 days/month)
    rolling_window_1y = 252    # 1 year (252 trading days)
    rolling_window_3y = 756    # 3 years (252 × 3 trading days)
    
    # Calculate rolling returns (as percentage values)
    rolling_data = filtered_data.copy()
    
    # Make sure we have enough data for calculation
    if len(rolling_data) > rolling_window_3m:
        rolling_data['rolling_3m_return'] = ((rolling_data['StrategyValue'] / rolling_data['StrategyValue'].shift(rolling_window_3m)) - 1) * 100
    else:
        rolling_data['rolling_3m_return'] = np.nan
        
    if len(rolling_data) > rolling_window_1y:
        rolling_data['rolling_1y_return'] = ((rolling_data['StrategyValue'] / rolling_data['StrategyValue'].shift(rolling_window_1y)) - 1) * 100
    else:
        rolling_data['rolling_1y_return'] = np.nan
        
    if len(rolling_data) > rolling_window_3y:
        rolling_data['rolling_3y_return'] = ((rolling_data['StrategyValue'] / rolling_data['StrategyValue'].shift(rolling_window_3y)) - 1) * 100
    else:
        rolling_data['rolling_3y_return'] = np.nan
    
    # Filter for non-NaN values to avoid issues
    rolling_data_3m = rolling_data.dropna(subset=['rolling_3m_return'])
    rolling_data_1y = rolling_data.dropna(subset=['rolling_1y_return'])
    rolling_data_3y = rolling_data.dropna(subset=['rolling_3y_return'])
    
    # Aggregate data on monthly basis
    rolling_monthly = pd.DataFrame()
    
    # Process each timeframe separately to avoid losing data
    if not rolling_data_3m.empty:
        rolling_monthly['rolling_3m_return'] = rolling_data_3m['rolling_3m_return'].resample('M').mean()
    
    if not rolling_data_1y.empty:
        rolling_monthly['rolling_1y_return'] = rolling_data_1y['rolling_1y_return'].resample('M').mean()
        
    if not rolling_data_3y.empty:
        rolling_monthly['rolling_3y_return'] = rolling_data_3y['rolling_3y_return'].resample('M').mean()
    
    # Function to create rolling returns plot
    def create_rolling_returns_plot(data, column, stats, color, period_name):
        # Create plot
        fig = go.Figure()
        
        # Main line for rolling returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=f'{period_name}-{t("rolling_returns").split()[0]}',
                line=dict(color=color, width=2.5)  # Increased line width
            )
        )
        
        # Horizontal line for average
        fig.add_trace(
            go.Scatter(
                x=[data.index.min(), data.index.max()],
                y=[stats['mean'], stats['mean']],
                mode='lines',
                name=t('average').format(round(stats["mean"], 2)),  # Increased precision
                line=dict(color='black', width=2, dash='dash')
            )
        )
        
        # Upper standard deviation
        fig.add_trace(
            go.Scatter(
                x=[data.index.min(), data.index.max()],
                y=[stats['mean'] + stats['std'], stats['mean'] + stats['std']],
                mode='lines',
                name=t('std_dev_plus').format(round(stats["mean"] + stats["std"], 2)),  # Increased precision
                line=dict(color='rgba(0, 128, 0, 0.5)', width=1.5, dash='dot')
            )
        )
        
        # Lower standard deviation
        fig.add_trace(
            go.Scatter(
                x=[data.index.min(), data.index.max()],
                y=[stats['mean'] - stats['std'], stats['mean'] - stats['std']],
                mode='lines',
                name=t('std_dev_minus').format(round(stats["mean"] - stats["std"], 2)),  # Increased precision
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1.5, dash='dot')
            )
        )
        
        # Maximum value with marker and label
        max_date = data[column].idxmax()
        max_value = data[column].max()
        fig.add_trace(
            go.Scatter(
                x=[max_date],
                y=[max_value],
                mode='markers+text',
                name=t('maximum').format(round(max_value, 2)),  # Increased precision
                marker=dict(color='green', size=12, symbol='triangle-up'),  # Larger marker
                text=f'{max_value:.2f}%',  # Increased precision
                textposition='top center',
                textfont=dict(color='green', size=12),  # Increased text size
                hoverinfo='text',
                hovertext=f'{t("maximum").split(":")[0]}: {max_date.strftime("%d.%m.%Y")}, {max_value:.2f}%'  # Increased precision
            )
        )
        
        # Minimum value with marker and label
        min_date = data[column].idxmin()
        min_value = data[column].min()
        fig.add_trace(
            go.Scatter(
                x=[min_date],
                y=[min_value],
                mode='markers+text',
                name=t('minimum').format(round(min_value, 2)),  # Increased precision
                marker=dict(color='red', size=12, symbol='triangle-down'),  # Larger marker
                text=f'{min_value:.2f}%',  # Increased precision
                textposition='bottom center',
                textfont=dict(color='red', size=12),  # Increased text size
                hoverinfo='text',
                hovertext=f'{t("minimum").split(":")[0]}: {min_date.strftime("%d.%m.%Y")}, {min_value:.2f}%'  # Increased precision
            )
        )
        
        # Determine padding for y-axis range to avoid cutoff
        y_padding = max(abs(max_value), abs(min_value)) * 0.15
        
        # Adjust layout
        fig.update_layout(
            title=t('rolling_returns_title').format(period_name, round(stats["mean"], 2), round(stats["std"], 2)),  # Increased precision
            xaxis_title=t('date'),
            yaxis_title=f'{period_name}-{t("return")}',
            height=600,  # Increased height
            autosize=True,
            margin=dict(l=50, r=50, t=100, b=50),  # Adjusted margins
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)  # Increased legend text size
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                gridcolor='lightgray',
                # Set y-axis range with padding to avoid cutoff
                range=[min_value - y_padding, max_value + y_padding],
                tickfont=dict(size=12)  # Increased axis tick font size
            ),
            xaxis=dict(
                tickfont=dict(size=12)  # Increased axis tick font size
            ),
            template="plotly_white"  # Use a clean white template
        )
        
        # Statistics annotation
        fig.add_annotation(
            x=0.01,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"{t('avg_return')} {stats['mean']:.2f}%<br>{t('std_dev')} {stats['std']:.2f}%<br>{t('min')} {stats['min']:.2f}%<br>{t('max')} {stats['max']:.2f}%",  # Increased precision
            showarrow=False,
            font=dict(
                family="Arial",
                size=14,  # Increased font size
                color="black"
            ),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",  # More opaque background
            bordercolor="black",
            borderwidth=1,
            borderpad=6  # Increased padding
        )
        
        return fig
    
    # Statistics for all available timeframes
    stats = {}
    timeframes = {
        '3m': {'window': rolling_window_3m, 'color': 'rgba(65, 105, 225, 0.8)', 'name': t('three_months')},
        '1y': {'window': rolling_window_1y, 'color': 'rgba(0, 128, 0, 0.8)', 'name': t('one_year')},
        '3y': {'window': rolling_window_3y, 'color': 'rgba(128, 0, 128, 0.8)', 'name': t('three_years')}
    }
    
    # Tabs for different timeframes inside the rolling returns tab
    roll_inner_tab_3m, roll_inner_tab_1y, roll_inner_tab_3y, roll_inner_tab_all = st.tabs([
        t('three_months'), 
        t('one_year'), 
        t('three_years'), 
        t('comparison')
    ])
    
    # Check data availability and calculate statistics
    has_data = {}
    for tf, info in timeframes.items():
        col_name = f'rolling_{tf}_return'
        has_data[tf] = col_name in rolling_monthly.columns and not rolling_monthly[col_name].dropna().empty
        
        if has_data[tf]:
            # Only use valid data
            valid_data = rolling_monthly[col_name].dropna()
            stats[tf] = {
                'mean': valid_data.mean(),
                'std': valid_data.std(),
                'min': valid_data.min(),
                'max': valid_data.max(),
                'data': valid_data
            }
    
    # Create individual plots for each period
    for tf, info in timeframes.items():
        tab = roll_inner_tab_3m if tf == '3m' else roll_inner_tab_1y if tf == '1y' else roll_inner_tab_3y
        
        if has_data[tf]:
            with tab:
                fig = create_rolling_returns_plot(
                    rolling_monthly, 
                    f'rolling_{tf}_return', 
                    stats[tf],
                    info['color'],
                    info['name']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button
                rolling_filename = f"rolling_{tf}_returns_{strategy_display_name_for_file}_{period_text}.png"
                get_image_download_link(fig, rolling_filename, f"📥 {t('download_rolling').format(info['name'])}")
        else:
            with tab:
                st.warning(t('not_enough_data').format(info['name'], info['window']))
    
    # Comparison plot for all available periods
    with roll_inner_tab_all:
        # At least one period must have data
        if any(has_data.values()):
            # Create combined plot
            fig_combined = go.Figure()
            
            # Add each available period
            for tf, info in timeframes.items():
                if has_data[tf]:
                    col_name = f'rolling_{tf}_return'
                    fig_combined.add_trace(
                        go.Scatter(
                            x=rolling_monthly.index,
                            y=rolling_monthly[col_name],
                            mode='lines',
                            name=f"{info['name']} (Ø: {stats[tf]['mean']:.2f}%)",  # Increased precision
                            line=dict(color=info['color'], width=2.5)  # Increased line width
                        )
                    )
            
            # Determine maximum and minimum values for y-axis
            max_y = -float('inf')
            min_y = float('inf')
            
            for tf in timeframes:
                if has_data[tf]:
                    # Get real min/max values from the data
                    col_data = rolling_monthly[f'rolling_{tf}_return'].dropna()
                    if not col_data.empty:
                        max_y = max(max_y, col_data.max())
                        min_y = min(min_y, col_data.min())
            
            # Add padding to avoid cutoff
            y_padding = max(abs(max_y), abs(min_y)) * 0.15
            
            # Adjust layout
            fig_combined.update_layout(
                title=t('comparison_title'),
                xaxis_title=t('date'),
                yaxis_title=t('return'),
                height=700,  # Increased height
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=50),  # Adjusted margins
                hovermode='x unified',
                yaxis=dict(
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black',
                    gridcolor='lightgray',
                    range=[min_y - y_padding, max_y + y_padding],
                    tickfont=dict(size=12)  # Increased axis tick font size
                ),
                xaxis=dict(
                    tickfont=dict(size=12)  # Increased axis tick font size
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)  # Increased legend text size
                ),
                template="plotly_white"  # Use a clean white template
            )
            
            # Statistics table as annotation
            table_html = f"<b>{t('statistics')}</b><br>"
            table_html += "<table style='width:100%'>"
            table_html += f"<tr><th>{t('period')}</th><th>{t('avg_return_pct')}</th><th>{t('std_dev_pct')}</th><th>{t('min_pct')}</th><th>{t('max_pct')}</th></tr>"
            
            for tf, info in timeframes.items():
                if has_data[tf]:
                    table_html += f"<tr><td>{info['name']}</td><td>{stats[tf]['mean']:.2f}%</td><td>{stats[tf]['std']:.2f}%</td><td>{stats[tf]['min']:.2f}%</td><td>{stats[tf]['max']:.2f}%</td></tr>"
            
            table_html += "</table>"
            
            fig_combined.add_annotation(
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                text=table_html,
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=14,  # Increased text size
                    color="black"
                ),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",  # More opaque background
                bordercolor="black",
                borderwidth=1,
                borderpad=6  # Increased padding
            )
            
            # Display chart
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Add download button
            rolling_filename = f"rolling_returns_comparison_{strategy_display_name_for_file}_{period_text}.png"
            get_image_download_link(fig_combined, rolling_filename, "📥 " + t('download_comparison'))
            
            # Additional data in a table
            st.subheader(t('detailed_statistics'))
            stats_df = pd.DataFrame({
                t('period'): [info['name'] for tf, info in timeframes.items() if has_data[tf]],
                t('avg_return_pct'): [stats[tf]['mean'] for tf in timeframes if has_data[tf]],
                t('std_dev_pct'): [stats[tf]['std'] for tf in timeframes if has_data[tf]],
                t('min_pct'): [stats[tf]['min'] for tf in timeframes if has_data[tf]],
                t('max_pct'): [stats[tf]['max'] for tf in timeframes if has_data[tf]]
            })
            st.dataframe(stats_df.set_index(t('period')).style.format({
                t('avg_return_pct'): '{:.2f}',  # Increased precision
                t('std_dev_pct'): '{:.2f}',     # Increased precision
                t('min_pct'): '{:.2f}',         # Increased precision
                t('max_pct'): '{:.2f}'          # Increased precision
            }))
        else:
            st.warning(t('not_enough_data').format(t('rolling_returns').split()[0], ""))
            
    st.markdown("</div>", unsafe_allow_html=True)
        
with details_tab:
    # Detailed metrics section
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📋 """ + t('detailed_analysis') + """</h3>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("##### " + t('performance_metrics'))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=t('cagr'), value=f"{cagr*100:.2f}%")
    with col2:
        st.metric(label=t('total_return'), value=f"{total_return*100:.2f}%")
    with col3:
        st.metric(label=t('volatility'), value=f"{annual_volatility*100:.2f}%")
    with col4:
        st.metric(label=t('max_drawdown'), value=f"{max_drawdown:.2f}%")
    
    # Risk metrics
    st.markdown("##### " + t('risk_metrics'))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=t('sharpe_ratio'), value=f"{sharpe:.2f}")
    with col2:
        st.metric(label=t('sortino_ratio'), value=f"{sortino:.2f}")
    with col3:
        st.metric(label=t('tracking_error'), value=f"{tracking_error*100:.2f}%")
    with col4:
        st.metric(label=t('information_ratio'), value=f"{information_ratio:.2f}")
    
    # Relative metrics
    st.markdown("##### " + t('relative_metrics'))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=t('beta'), value=f"{beta:.2f}")
    with col2:
        st.metric(label=t('alpha'), value=f"{alpha_annualized_pct:.2f}%")
    with col3:
        st.metric(label=t('avg_recovery'), value=f"{avg_recovery:.0f} {t('days')}")
    with col4:
        st.metric(label=t('max_recovery'), value=f"{max_recovery:.0f} {t('days')}")
        
    # Period Performance
    st.markdown("##### " + t('period_performance'))
    
    # Create DataFrame for better presentation
    period_data = pd.DataFrame({
        t('period'): [t('ytd'), t('one_year_perf'), t('three_year_perf'), t('five_year_perf')],
        t('strategy'): [
            f"{ytd_perf:.2f}%", 
            f"{one_year_perf:.2f}%", 
            f"{three_year_perf:.2f}%", 
            f"{five_year_perf:.2f}%"
        ],
        t('benchmark_label'): [
            f"{ytd_bench_perf:.2f}%", 
            f"{one_year_bench_perf:.2f}%", 
            f"{three_year_bench_perf:.2f}%", 
            f"{five_year_bench_perf:.2f}%"
        ],
        'Difference': [
            f"{ytd_perf - ytd_bench_perf:.2f}%", 
            f"{one_year_perf - one_year_bench_perf:.2f}%", 
            f"{three_year_perf - three_year_bench_perf:.2f}%", 
            f"{five_year_perf - five_year_bench_perf:.2f}%"
        ]
    })
    
    # Display the performance comparison table
    st.dataframe(period_data.set_index(t('period')), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer with credits
st.markdown("""
<div style="text-align: center; margin-top: 20px; padding: 10px; color: #666;">
    <p>© 2024</p>
</div>
""", unsafe_allow_html=True) 
