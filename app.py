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
import seaborn as sns
import matplotlib.gridspec as gridspec

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

# Use path that works both locally and in cloud
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set page configuration
st.set_page_config(
    page_title="Quantmade AI Quant Funds Strategy Dashboard",
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
        background: linear-gradient(to bottom, #fff, #f9f9fa);
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #e6e6e6;
        display: flex;
        flex-direction: column;
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }
    
    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 3px;
    }
    
    .metric-label {
        font-size: 11px;
        color: #555;
        font-weight: 500;
        margin-bottom: 0;
    }
    
    .metric-icon {
        color: #777;
        font-size: 10px;
        padding: 2px;
        background: #f0f5fa;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        line-height: 1.2;
        display: flex;
        align-items: center;
        padding-top: 4px;
        padding-bottom: 2px;
        margin-top: auto;
    }
    
    .positive-value {
        color: #0fa76f;
    }
    
    .negative-value {
        color: #e05260;
    }
    
    .trend-indicator {
        font-size: 12px;
        margin-right: 2px;
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
    
    /* Add modern card-based styling for metrics */
    .kpi-grid-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    
    .kpi-card {
        background-color: white;
        border-radius: 5px;
        padding: 10px 8px;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 1px 3px 0px, rgba(0, 0, 0, 0.06) 0px 1px 2px 0px;
        border: 1px solid #eaeaea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        aspect-ratio: 1/1;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 6px -1px, rgba(0, 0, 0, 0.06) 0px 2px 4px -1px;
    }
    
    .kpi-category {
        font-size: 8px;
        font-weight: 500;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 3px;
    }
    
    .kpi-title {
        font-size: 11px;
        font-weight: 600;
        color: #333;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
    }
    
    .kpi-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        font-size: 11px;
    }
    
    .kpi-value-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    .kpi-value {
        font-size: 20px;
        font-weight: 700;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2px 0;
    }
    
    .performance {
        background-color: #f0f7ff;
        border-left: 3px solid #0066ff;
    }
    
    .risk {
        background-color: #fff5f5;
        border-left: 3px solid #ff4d4f;
    }
    
    .ratio {
        background-color: #f6ffed;
        border-left: 3px solid #52c41a;
    }
    
    .positive-value {
        color: #52c41a;
    }
    
    .negative-value {
        color: #ff4d4f;
    }
    
    .trend-indicator {
        font-size: 12px;
        margin-right: 2px;
    }
    
    .kpi-change {
        font-size: 12px;
        color: #666;
    }
    
    .metric-section {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 3px 5px;
        margin-bottom: 20px;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Period performance cards */
    .period-performance-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        grid-gap: 16px;
        margin-top: 10px;
    }
    
    .period-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 14px;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 1px 3px 0px, rgba(0, 0, 0, 0.06) 0px 1px 2px 0px;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .period-card:hover {
        transform: translateY(-2px);
        box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 6px -1px, rgba(0, 0, 0, 0.06) 0px 2px 4px -1px;
    }
    
    .period-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #333;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 5px;
    }
    
    .compare-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    
    .entity-label {
        color: #555;
    }
    
    /* Compact the info box */
    .info-box {
        background-color: #f8fafc;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.85em;
        color: #334155;
        line-height: 1.4;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
        border: 1px solid #e2e8f0;
    }
    
    /* Add compact date info */
    .date-info {
        background-color: #f8fafc;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.85em;
        line-height: 1.4;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
        border: 1px solid #e2e8f0;
    }
    
    /* Make tab content more condensed */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 16px;
    }
    
    /* KPI table styling */
    .kpi-table-container {
        padding: 16px;
        background-color: white;
        border-radius: 8px;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
        margin-bottom: 20px;
    }
    
    .kpi-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    
    .kpi-table th {
        background-color: #f8f9fa;
        color: #555;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
        border-bottom: 2px solid #e9ecef;
    }
    
    .kpi-table tr {
        border-bottom: 1px solid #e9ecef;
    }
    
    .kpi-table tr:last-child {
        border-bottom: none;
    }
    
    .kpi-table td {
        padding: 10px 15px;
        vertical-align: middle;
    }
    
    .kpi-category {
        font-weight: 600;
        font-size: 14px;
        color: #333;
    }
    
    .kpi-label {
        padding-left: 10px;
    }
    
    .kpi-value {
        text-align: right;
        font-weight: 700;
        font-size: 16px;
    }
    
    .positive-value {
        color: #52c41a;
    }
    
    .negative-value {
        color: #ff4d4f;
    }
    
    .trend-indicator {
        font-size: 12px;
        margin-right: 3px;
    }
    
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }
    
    .simple-table th {
        background-color: #f5f5f5;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    
    .simple-table td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    
    .positive {
        color: green;
    }
    
    .negative {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Pre-selected quants
selected_quant_files = ["STEADY US 100performance.csv", "STEADY US Tech 100performance.csv"]
selected_quant_names = ["STEADY US 100", "STEADY US Tech 100"]
risk_free_rate = 2.0 / 100  # Default 2.0%

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
        'return_analysis': 'Return Analysis',
        'benchmark': 'Equal Weight Benchmark (50/50)'
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
        'return_analysis': 'Rendite-Analyse',
        'benchmark': 'Equal Weight Benchmark (50/50)'
    }
}

# Session state for language
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

# Set up weights (evenly distributed)
weights = {}
for quant_name in selected_quant_names:
    weights[quant_name] = 0.5  # 50% each

# Function to load data from any available path
def load_data(filename, display_name):
    possible_paths = [
        os.path.join(base_dir, 'data', 'PerformancesClean', filename),
        os.path.join(base_dir, 'data', 'Performances', filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                return df
            except Exception as e:
                st.error(f"Error loading {path}: {str(e)}")
    
    # If no file found, create dummy data
    st.warning(f"Creating dummy data for {display_name}")
    date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    df = pd.DataFrame(index=date_range)
    df['Returns'] = np.random.normal(0.0003, 0.005, len(date_range))
    df['Benchmark'] = 100 * (1 + np.random.normal(0.0002, 0.006, len(date_range))).cumprod()
    return df

# Process data for selected quants
returns_df = pd.DataFrame()
benchmark_returns_df = pd.DataFrame()

# Load data for each quant
for quant_file, display_name in zip(selected_quant_files, selected_quant_names):
    df = load_data(quant_file, display_name)
    returns_df[display_name] = df["Returns"]
    benchmark_returns_df[display_name] = df["Benchmark"].pct_change()

# Fill NaN values
returns_df.fillna(0, inplace=True)
benchmark_returns_df.fillna(0, inplace=True)

# Check if data was loaded
if returns_df.empty:
    st.error("No data was loaded. Please check file paths.")
    st.stop()

# Calculate weighted portfolio returns and create 50/50 blended benchmark
quant_data = pd.DataFrame(index=returns_df.index)
quant_data["Returns"] = 0

# Create 50/50 blend of benchmarks (each strategy contributes its own benchmark at 50%)
# The benchmarks in the CSV files should already be the equal-weight indices
quant_data["Benchmark_Returns"] = 0
for quant_name in selected_quant_names:
    # Strategy returns with weights
    quant_data["Returns"] += returns_df[quant_name] * weights[quant_name]
    # Benchmark returns with weights - these are now equal-weight benchmark returns
    quant_data["Benchmark_Returns"] += benchmark_returns_df[quant_name] * weights[quant_name]

# Calculate additional metrics
quant_data["StrategyValue"] = 100 * (1 + quant_data["Returns"]).cumprod()
quant_data["Benchmark"] = 100 * (1 + quant_data["Benchmark_Returns"]).cumprod()
quant_data["drawdown"] = -((quant_data["StrategyValue"].cummax() - quant_data["StrategyValue"])/quant_data["StrategyValue"].cummax())*100
quant_data["benchmark_drawdown"] = -((quant_data["Benchmark"].cummax() - quant_data["Benchmark"])/quant_data["Benchmark"].cummax())*100
quant_data["monthly_returns"] = (1 + quant_data["Returns"]).resample("M").prod() - 1

# Filter data to start from 2015 at the latest
max_start_date = pd.Timestamp('2015-01-01')
if quant_data.index.min() < max_start_date:
    # If data goes back before 2015, limit it to start from 2015
    quant_data = quant_data.loc[max_start_date:].copy()
    
    # Recalibrate to 100 after filtering
    first_strategy_value = quant_data["StrategyValue"].iloc[0]
    first_benchmark_value = quant_data["Benchmark"].iloc[0]
    
    quant_data["StrategyValue"] = quant_data["StrategyValue"] / first_strategy_value * 100
    quant_data["Benchmark"] = quant_data["Benchmark"] / first_benchmark_value * 100
    
    # Recalculate drawdowns after rescaling
    quant_data["drawdown"] = -((quant_data["StrategyValue"].cummax() - quant_data["StrategyValue"])/quant_data["StrategyValue"].cummax())*100
    quant_data["benchmark_drawdown"] = -((quant_data["Benchmark"].cummax() - quant_data["Benchmark"])/quant_data["Benchmark"].cummax())*100

# Dashboard header with strategy name
strategy_display_name = " + ".join(selected_quant_names)
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <h1>{t('dashboard_title')}</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# Date selection in an elegant container
st.markdown(f"""<div class="date-selector-container"><h4>📅 {t('time_period')}</h4>""", unsafe_allow_html=True)

# Date selector variables
min_date = quant_data.index.min().date()
max_date = quant_data.index.max().date()

# Time period buttons
period_options = ["YTD", t('one_year'), "3 " + t('three_years').split()[-1], "5 " + t('three_years').split()[-1], "MAX"]
selected_period = st.radio(t('select_period'), options=period_options, horizontal=True, index=4)  # Default to MAX

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

# Always recalibrate to 100 at the beginning of the selected period
# Scale data to 100, calculate drawdown after scaling
first_value = filtered_data["StrategyValue"].iloc[0]
first_benchmark = filtered_data["Benchmark"].iloc[0]

filtered_data["StrategyValue"] = filtered_data["StrategyValue"] / first_value * 100
filtered_data["Benchmark"] = filtered_data["Benchmark"] / first_benchmark * 100
filtered_data["Cummax"] = filtered_data["StrategyValue"].cummax()
filtered_data["drawdown"] = -((filtered_data["Cummax"] - filtered_data["StrategyValue"])/filtered_data["Cummax"])*100
filtered_data["benchmark_drawdown"] = -((filtered_data["Benchmark"].cummax() - filtered_data["Benchmark"])/filtered_data["Benchmark"].cummax())*100
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

# Initialize period performances with zeros
ytd_perf, one_year_perf, three_year_perf, five_year_perf = 0, 0, 0, 0
ytd_bench_perf, one_year_bench_perf, three_year_bench_perf, five_year_bench_perf = 0, 0, 0, 0

# Strategy performance
if ytd_start_date != last_date:
    # Create a temporary scaled dataframe for each period to always start at 100
    ytd_data = quant_data.loc[ytd_start_date:last_date].copy()
    first_value_ytd = ytd_data["StrategyValue"].iloc[0]
    first_bench_ytd = ytd_data["Benchmark"].iloc[0]
    ytd_data["StrategyValue"] = ytd_data["StrategyValue"] / first_value_ytd * 100
    ytd_data["Benchmark"] = ytd_data["Benchmark"] / first_bench_ytd * 100
    ytd_perf = (ytd_data["StrategyValue"].iloc[-1] / 100 - 1) * 100
    ytd_bench_perf = (ytd_data["Benchmark"].iloc[-1] / 100 - 1) * 100

if one_year_date != last_date:
    # One year period
    one_year_data = quant_data.loc[one_year_date:last_date].copy()
    first_value_1y = one_year_data["StrategyValue"].iloc[0]
    first_bench_1y = one_year_data["Benchmark"].iloc[0]
    one_year_data["StrategyValue"] = one_year_data["StrategyValue"] / first_value_1y * 100
    one_year_data["Benchmark"] = one_year_data["Benchmark"] / first_bench_1y * 100
    one_year_perf = (one_year_data["StrategyValue"].iloc[-1] / 100 - 1) * 100
    one_year_bench_perf = (one_year_data["Benchmark"].iloc[-1] / 100 - 1) * 100

if three_year_date != last_date:
    # Three year period
    three_year_data = quant_data.loc[three_year_date:last_date].copy()
    first_value_3y = three_year_data["StrategyValue"].iloc[0]
    first_bench_3y = three_year_data["Benchmark"].iloc[0]
    three_year_data["StrategyValue"] = three_year_data["StrategyValue"] / first_value_3y * 100
    three_year_data["Benchmark"] = three_year_data["Benchmark"] / first_bench_3y * 100
    three_year_perf = (three_year_data["StrategyValue"].iloc[-1] / 100 - 1) * 100
    three_year_bench_perf = (three_year_data["Benchmark"].iloc[-1] / 100 - 1) * 100

if five_year_date != last_date:
    # Five year period
    five_year_data = quant_data.loc[five_year_date:last_date].copy()
    first_value_5y = five_year_data["StrategyValue"].iloc[0]
    first_bench_5y = five_year_data["Benchmark"].iloc[0]
    five_year_data["StrategyValue"] = five_year_data["StrategyValue"] / first_value_5y * 100
    five_year_data["Benchmark"] = five_year_data["Benchmark"] / first_bench_5y * 100
    five_year_perf = (five_year_data["StrategyValue"].iloc[-1] / 100 - 1) * 100
    five_year_bench_perf = (five_year_data["Benchmark"].iloc[-1] / 100 - 1) * 100

# Create main dashboard tabs
kpi_tab, performance_tab, returns_tab, rolling_tab, details_tab = st.tabs([
    "📊 " + t('key_metrics'),
    "📈 " + t('performance_drawdown'),
    "💹 " + t('return_analysis'),
    "📈 " + t('rolling_returns'),
    "📋 " + t('detailed_analysis')
])

# KPI TAB - Key metrics
with kpi_tab:
    # Add simplified table styling
    st.markdown("""
    <style>
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }
    
    .simple-table th {
        background-color: #f5f5f5;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    
    .simple-table td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    
    .positive {
        color: green;
    }
    
    .negative {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display calculation period
    st.markdown(f"**Calculation Period:** {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
    
    # Create a simple table for KPIs
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Performance Metrics")
        performance_data = {
            "Metric": ["CAGR", "YTD", "1Y Return", "3Y Return"],
            "Value": [
                f"{cagr*100:.2f}%" if cagr >= 0 else f"<span class='negative'>{cagr*100:.2f}%</span>",
                f"{ytd_perf:.2f}%" if ytd_perf >= 0 else f"<span class='negative'>{ytd_perf:.2f}%</span>",
                f"{one_year_perf:.2f}%" if one_year_perf >= 0 else f"<span class='negative'>{one_year_perf:.2f}%</span>",
                f"{three_year_perf:.2f}%" if three_year_perf >= 0 else f"<span class='negative'>{three_year_perf:.2f}%</span>"
            ]
        }
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(performance_data)
        st.write(performance_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)
    
    # Risk metrics
    with col2:
        st.header("Risk Metrics")
        risk_data = {
            "Metric": ["Volatility", "Max Drawdown", "Tracking Error"],
            "Value": [
                f"{annual_volatility*100:.2f}%",
                f"{max_drawdown:.2f}%",
                f"{tracking_error*100:.2f}%"
            ]
        }
    
        # Convert to DataFrame
        risk_df = pd.DataFrame(risk_data)
        st.write(risk_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)
    
    # Ratio metrics
    with col3:
        st.header("Ratio Metrics")
        ratio_data = {
            "Metric": ["Sharpe Ratio", "Sortino Ratio", "Beta", "Alpha"],
            "Value": [
                f"{sharpe:.2f}" if sharpe >= 0 else f"<span class='negative'>{sharpe:.2f}</span>",
                f"{sortino:.2f}" if sortino >= 0 else f"<span class='negative'>{sortino:.2f}</span>",
                f"{beta:.2f}" if beta >= 0 else f"<span class='negative'>{beta:.2f}</span>",
                f"{alpha_annualized_pct:.2f}" if alpha_annualized_pct >= 0 else f"<span class='negative'>{alpha_annualized_pct:.2f}</span>"
            ]
        }
        
        # Convert to DataFrame
        ratio_df = pd.DataFrame(ratio_data)
        st.write(ratio_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)

# PERFORMANCE TAB - Performance and Drawdown chart
with performance_tab:
    # Replace the date info with more compact version
    date_info_html = f"""
    <div class="date-info">
        <strong>Calculation Period:</strong> {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}
    </div>
    """
    st.markdown(date_info_html, unsafe_allow_html=True)
    
    # Performance and Drawdown visualization
    st.markdown("""<div class="chart-container">
        <h3 class="metric-section-header">📈 """ + t('performance_drawdown') + """</h3>
    """, unsafe_allow_html=True)

    # Create subplot with two charts - remove subplot titles to avoid "undefined" text
    fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.03,  # Decreased for better spacing in taller chart
                    row_heights=[0.7, 0.3])  # Adjusted for better proportion with taller chart

    # Strategy value plot with area filling
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index, 
            y=filtered_data['StrategyValue'],
            mode='lines',
            name=t('strategy_value'),
            line=dict(color='rgba(65, 105, 225, 0.8)', width=1),  # Increased line width
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
            line=dict(color='rgba(255, 85, 0, 0.8)', width=1),  # Increased line width
            fill='tozeroy',
            fillcolor='rgba(255, 85, 0, 0.1)'
        ),
        row=1, col=1
    )

    # Drawdown plots
    # Strategy drawdown
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['drawdown'],
            mode='lines',
            name=t('strategy') + " " + t('drawdown'),
            line=dict(color='rgba(65, 105, 225, 0.8)', width=1),  # Increased line width
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.1)'
        ),
        row=2, col=1
    )
    
    # Benchmark drawdown
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['benchmark_drawdown'],
            mode='lines',
            name=t('benchmark_label') + " " + t('drawdown'),
            line=dict(color='rgba(255, 85, 0, 0.8)', width=1),  # Increased line width
            fill='tozeroy',
            fillcolor='rgba(255, 85, 0, 0.1)'
        ),
        row=2, col=1
    )

    # Adjust layout with larger font sizes and increased height and margins
    fig.update_layout(
        height=900,  # Increased height by factor of 1.5 (from 600 to 900)
        showlegend=True,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),  # Adjusted margins
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(size=16)  # Increased legend text size
        ),
        template="plotly_white"  # Use a clean white template
    )

    # Add explicit titles to each subplot instead of using subplot_titles
    fig.update_yaxes(
        title_text=t('strategy_value'),
        title_font=dict(size=16),
        tickfont=dict(size=16),
        row=1, 
        col=1
    )
    
    fig.update_yaxes(
        title_text=t('drawdown'),
        title_font=dict(size=16),
        tickfont=dict(size=16),
        row=2, 
        col=1
    )
    
    # X-axis between plots
    fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide X-axis in upper plot 
    
    fig.update_xaxes(
        title_text=t('date'),
        title_font=dict(size=16),
        tickfont=dict(size=16),
        title_standoff=5,
        row=2, 
        col=1
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button for performance chart with higher resolution
    strategy_name = "_".join(selected_quant_files) if len(selected_quant_files) > 1 else selected_quant_files[0]
    strategy_display_name_for_file = "_".join(selected_quant_names)
    period_text = selected_period if selected_period else f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    performance_filename = f"performance_drawdown_{strategy_display_name_for_file}_{period_text}.png"
    
    # Increased width, height, and scale for higher resolution download
    get_image_download_link(fig, performance_filename, "📥 " + t('download_performance'), 
                           width=2400, height=1600, scale=3)
    
    # Close chart container
    st.markdown("</div>", unsafe_allow_html=True)

# RETURNS TAB - Monthly returns analysis
with returns_tab:
    # Replace the date info with more compact version
    date_info_html = f"""
    <div class="date-info">
        <strong>Calculation Period:</strong> {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}
    </div>
    """
    st.markdown(date_info_html, unsafe_allow_html=True)
    
    st.markdown("""<div class="chart-container">
        <h3 class="metric-section-header">📅 """ + t('monthly_returns').format(start_date.strftime("%d.%m.%Y")) + """</h3>
    """, unsafe_allow_html=True)
    
    try:
        # Use a Plotly approach instead of seaborn
        import plotly.graph_objects as go
        
        # Set start date to January 1, 2020 (regardless of the selected start date in the dashboard)
        start_date_for_heatmap = pd.Timestamp('2020-01-01')
        returns_data = filtered_data.loc[start_date_for_heatmap:, 'Returns'].copy()
        
        # Add note about specific date range for heatmap
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 8px 15px; border-radius: 5px; margin-bottom: 15px; font-size: 0.85em; color: #0c5460;">
            <strong>Note:</strong> Monthly returns displayed from {start_date_for_heatmap.strftime('%d.%m.%Y')} to {end_date.strftime('%d.%m.%Y')} regardless of the selected period.
        </div>
        """, unsafe_allow_html=True)
        
        # Create a DataFrame with Year and Month
        returns_df = pd.DataFrame({
            'Return': returns_data,
            'Year': returns_data.index.year,
            'Month': returns_data.index.month,
            'Date': returns_data.index
        })
        
        # Calculate monthly returns by grouping
        monthly_returns = returns_df.groupby(['Year', 'Month'])['Return'].apply(
            lambda x: ((1 + x).prod() - 1) * 100  # Convert to percentage
        ).reset_index()
        
        # Create pivot table for heatmap
        pivot_data = monthly_returns.pivot(index='Year', columns='Month', values='Return')
        
        # Get current year and month for highlighting
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        # Replace month numbers with names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mär' if st.session_state.language == 'de' else 'Mar', 
            4: 'Apr', 5: 'Mai' if st.session_state.language == 'de' else 'May', 
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 
            10: 'Okt' if st.session_state.language == 'de' else 'Oct', 
            11: 'Nov', 12: 'Dez' if st.session_state.language == 'de' else 'Dec'
        }
        
        # Create lists for the heatmap
        years = pivot_data.index.tolist()
        months = [month_names[m] for m in pivot_data.columns.tolist()]
        z_values = pivot_data.values
        
        # Create text annotations for the heatmap cells
        text_values = []
        for row in z_values:
            row_texts = []
            for value in row:
                if pd.notna(value):
                    sign = "+" if value > 0 else ""
                    row_texts.append(f"{sign}{value:.1f}%")
                else:
                    row_texts.append("")
            text_values.append(row_texts)
        
        # Determine the color scale range
        abs_max = max(abs(np.nanmin(z_values)), abs(np.nanmax(z_values)))
        
        # Create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=months,
            y=years,
            colorscale=[
                [0, 'rgb(165,0,38)'],      # Dark red for very negative
                [0.3, 'rgb(215,48,39)'],   # Red for negative
                [0.45, 'rgb(244,109,67)'], # Light red-orange
                [0.5, 'rgb(255,255,255)'], # White for zero
                [0.55, 'rgb(186,228,174)'],# Light green for slightly positive
                [0.65, 'rgb(152,210,144)'],# Light green
                [0.8, 'rgb(88,171,97)'],   # Green for positive
                [1, 'rgb(35,120,35)']      # Dark green for very positive
            ],
            colorbar=dict(
                title=dict(text=t('return') + " (%)", side="right"),
                thickness=12,
                len=0.8,
                x=1.02
            ),
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(
                family="Arial", 
                size=12,
                color="black"  # Use a single color for all text
            ),
            hoverongaps=False,
            hovertemplate='%{y}, %{x}: %{text}<extra></extra>',
            zmid=0,  # Center the color scale at zero
            zmin=-abs_max,
            zmax=abs_max
        ))
        
        # Now add colored text annotations separately
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                if j < len(months) and i < len(years) and j < z_values.shape[1] and pd.notna(z_values[i, j]):
                    value = z_values[i, j]
                    color = "darkgreen" if value > 0 else "darkred" if value < 0 else "black"
                    
                    # Create an annotation object
                    fig.add_annotation(
                        x=month,
                        y=year,
                        text=text_values[i][j],
                        showarrow=False,
                        font=dict(
                            family="Arial",
                            size=12,
                            color=color
                        ),
                        xref="x",
                        yref="y"
                    )
        
        # Improve the layout
        fig.update_layout(
            title=f"Monthly Returns ({start_date_for_heatmap.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})",
            title_font=dict(size=16),
            width=700,
            height=400,
            margin=dict(l=40, r=40, t=60, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                title=None,
                side='top',
                tickangle=0,
                tickfont=dict(size=24)
            ),
            yaxis=dict(
                title=None,
                autorange="reversed",  # To keep newest years at top
                tickfont=dict(size=24)
            )
        )
        
        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a download option using the get_image_download_link function
        heatmap_filename = f"monthly_returns_{strategy_display_name_for_file}_{period_text}.png"
        get_image_download_link(fig, heatmap_filename, "📥 " + t('download_heatmap'))
        
        # Add explanatory text
        st.markdown("""
        <div style="font-size: 0.9em; color: #666;">
        <p><b>Reading the heatmap:</b></p>
        <ul>
            <li>The table shows monthly returns as a percentage.</li>
            <li>Green indicates positive returns, red indicates negative returns.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error creating monthly returns heatmap: {str(e)}")
        st.write("Try selecting a different date range.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ROLLING RETURNS TAB - Rolling returns analysis
with rolling_tab:
    # Replace the date info with more compact version
    date_info_html = f"""
    <div class="date-info">
        <strong>Calculation Period:</strong> {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}
    </div>
    """
    st.markdown(date_info_html, unsafe_allow_html=True)
    
    st.markdown("""<div class="chart-container">
        <h3 class="metric-section-header">📈 """ + t('rolling_returns') + """</h3>
    """, unsafe_allow_html=True)
    
    # Create tabs for different rolling periods
    rolling_tabs = st.tabs([
        "📊 " + t('three_months'),
        "📈 " + t('one_year'),
        "📉 " + t('three_years'),
        "🔄 " + t('comparison')
    ])
    
    # Calculate rolling returns for different periods
    rolling_data_3m = pd.DataFrame()
    rolling_data_1y = pd.DataFrame()
    rolling_data_3y = pd.DataFrame()
    
    # 3-Month rolling returns (approx. 63 trading days)
    window_size_3m = 63
    if len(filtered_data) >= window_size_3m:
        rolling_data_3m['rolling_3m_return'] = (filtered_data['StrategyValue'].pct_change(window_size_3m) * 100).dropna()
    
    # 1-Year rolling returns (approx. 252 trading days)
    window_size_1y = 252
    if len(filtered_data) >= window_size_1y:
        rolling_data_1y['rolling_1y_return'] = (filtered_data['StrategyValue'].pct_change(window_size_1y) * 100).dropna()
    
    # 3-Year rolling returns (approx. 756 trading days)
    window_size_3y = 756
    if len(filtered_data) >= window_size_3y:
        rolling_data_3y['rolling_3y_return'] = (filtered_data['StrategyValue'].pct_change(window_size_3y) * 100).dropna()
    
    # Function to create rolling returns plot
    def create_rolling_returns_plot(data, column, color, period_name, window_size):
        if data.empty:
            st.warning(t('not_enough_data').format(period_name, window_size))
            return None
        
        # Calculate statistics
        avg_return = data[column].mean()
        std_dev = data[column].std()
        min_return = data[column].min()
        max_return = data[column].max()
        
        # Create plot with increased dimensions
        fig = go.Figure()
        
        # Main rolling returns line with increased width
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[column],
                mode='lines',
                name=f"{period_name} {t('rolling_returns')}",
                line=dict(color=color, width=1.5)  # Increased line width
            )
        )
        
        # Average line shape
        fig.add_shape(
            type="line",
            x0=data.index.min(),
            y0=avg_return,
            x1=data.index.max(),
            y1=avg_return,
            line=dict(color="green", width=1, dash="dash")  # Increased line width
        )
        
        # Standard deviation bands shapes with increased width
        fig.add_shape(
            type="line",
            x0=data.index.min(),
            y0=avg_return + std_dev,
            x1=data.index.max(),
            y1=avg_return + std_dev,
            line=dict(color="gray", width=1, dash="dot")  # Increased line width
        )
        
        fig.add_shape(
            type="line",
            x0=data.index.min(),
            y0=avg_return - std_dev,
            x1=data.index.max(),
            y1=avg_return - std_dev,
            line=dict(color="gray", width=1, dash="dot")  # Increased line width
        )
        
        # Zusätzliche Traces für die Legende
        # Durchschnittslinie
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name="average",
                line=dict(color="green", width=1, dash="dash")
            )
        )
        
        # Standardabweichung +1σ
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name="+1σ",
                line=dict(color="gray", width=1, dash="dot")
            )
        )
        
        # Standardabweichung -1σ
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name="-1σ",
                line=dict(color="gray", width=1, dash="dot")
            )
        )
        
        # Statt Annotations eine Legendenbox für Statistiken hinzufügen
        # Leere Liste für Annotations, da wir jetzt die Legende verwenden
        annotations = []
        
        # Adjust layout with increased dimensions and font sizes
        fig.update_layout(
            title=t('rolling_returns_title').format(period_name, avg_return, std_dev),
            xaxis_title=t('date'),
            yaxis_title=t('return'),
            height=600,  # Increased height
            annotations=annotations,
            template="plotly_white",
            hovermode="x unified",
            title_font=dict(size=36),  # Increased title font size
            margin=dict(l=60, r=60, t=100, b=60),  # Increased margins
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=16)
            )
        )
        
        # Increase axis font sizes
        fig.update_xaxes(
            title_font=dict(size=24),
            tickfont=dict(size=24)
        )
        
        fig.update_yaxes(
            title_font=dict(size=24),
            tickfont=dict(size=24)
        )
        
        return fig, {
            'avg': avg_return,
            'std': std_dev,
            'min': min_return,
            'max': max_return
        }
    
    # 3-Month Rolling Returns Tab
    with rolling_tabs[0]:
        if not rolling_data_3m.empty:
            fig_3m, stats_3m = create_rolling_returns_plot(rolling_data_3m, 'rolling_3m_return', 'rgba(65, 105, 225, 0.8)', '3 ' + t('three_months').split()[-1], window_size_3m)
            if fig_3m:
                st.plotly_chart(fig_3m, use_container_width=True)
                filename_3m = f"rolling_3m_{strategy_display_name_for_file}_{period_text}.png"
                get_image_download_link(fig_3m, filename_3m, "📥 " + t('download_rolling').format('3 ' + t('three_months').split()[-1]))
                
                # Show statistics
                # Erstelle eine Tabelle für die Statistiken im gleichen Format wie die anderen Tabellen
                stats_data = {
                    "Metric": ["Durchschnitt", "Standardabweichung", "Minimum", "Maximum"],
                    "Value": [
                        f"{stats_3m['avg']:.2f}%",
                        f"{stats_3m['std']:.2f}%",
                        f"{stats_3m['min']:.2f}%",
                        f"{stats_3m['max']:.2f}%"
                    ]
                }
                
                # Konvertiere zu DataFrame und zeige als HTML-Tabelle an
                stats_df = pd.DataFrame(stats_data)
                st.write(stats_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)
        else:
            st.warning(t('not_enough_data').format('3 ' + t('three_months').split()[-1], window_size_3m))
    
    # 1-Year Rolling Returns Tab
    with rolling_tabs[1]:
        if not rolling_data_1y.empty:
            fig_1y, stats_1y = create_rolling_returns_plot(rolling_data_1y, 'rolling_1y_return', 'rgba(255, 102, 0, 0.8)', t('one_year'), window_size_1y)
            if fig_1y:
                st.plotly_chart(fig_1y, use_container_width=True)
                filename_1y = f"rolling_1y_{strategy_display_name_for_file}_{period_text}.png"
                get_image_download_link(fig_1y, filename_1y, "📥 " + t('download_rolling').format('1 ' + t('one_year')))
                
                # Show statistics
                # Erstelle eine Tabelle für die Statistiken im gleichen Format wie die anderen Tabellen
                stats_data = {
                    "Metric": ["Durchschnitt", "Standardabweichung", "Minimum", "Maximum"],
                    "Value": [
                        f"{stats_1y['avg']:.2f}%",
                        f"{stats_1y['std']:.2f}%",
                        f"{stats_1y['min']:.2f}%",
                        f"{stats_1y['max']:.2f}%"
                    ]
                }
                
                # Konvertiere zu DataFrame und zeige als HTML-Tabelle an
                stats_df = pd.DataFrame(stats_data)
                st.write(stats_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)
        else:
            st.warning(t('not_enough_data').format('1 ' + t('one_year'), window_size_1y))
    
    # 3-Year Rolling Returns Tab
    with rolling_tabs[2]:
        if not rolling_data_3y.empty:
            fig_3y, stats_3y = create_rolling_returns_plot(rolling_data_3y, 'rolling_3y_return', 'rgba(0, 153, 0, 0.8)', '3 ' + t('three_years').split()[-1], window_size_3y)
            if fig_3y:
                st.plotly_chart(fig_3y, use_container_width=True)
                filename_3y = f"rolling_3y_{strategy_display_name_for_file}_{period_text}.png"
                get_image_download_link(fig_3y, filename_3y, "📥 " + t('download_rolling').format('3 ' + t('three_years').split()[-1]))
                
                # Show statistics
                # Erstelle eine Tabelle für die Statistiken im gleichen Format wie die anderen Tabellen
                stats_data = {
                    "Metric": ["Durchschnitt", "Standardabweichung", "Minimum", "Maximum"],
                    "Value": [
                        f"{stats_3y['avg']:.2f}%",
                        f"{stats_3y['std']:.2f}%",
                        f"{stats_3y['min']:.2f}%",
                        f"{stats_3y['max']:.2f}%"
                    ]
                }
                
                # Konvertiere zu DataFrame und zeige als HTML-Tabelle an
                stats_df = pd.DataFrame(stats_data)
                st.write(stats_df.to_html(escape=False, index=False, classes='simple-table'), unsafe_allow_html=True)
        else:
            st.warning(t('not_enough_data').format('3 ' + t('three_years').split()[-1], window_size_3y))

    # Comparison Tab
    with rolling_tabs[3]:
        # Create DataFrame for monthly aggregated rolling returns
        rolling_monthly = pd.DataFrame()
        has_data = False
        
        if not rolling_data_3m.empty:
            rolling_monthly['rolling_3m_return'] = rolling_data_3m['rolling_3m_return'].resample('ME').mean()
            has_data = True
        
        if not rolling_data_1y.empty:
            rolling_monthly['rolling_1y_return'] = rolling_data_1y['rolling_1y_return'].resample('ME').mean()
            has_data = True
        
        if not rolling_data_3y.empty:
            rolling_monthly['rolling_3y_return'] = rolling_data_3y['rolling_3y_return'].resample('ME').mean()
            has_data = True
        
        if has_data:
            # Only use column names that actually exist in the DataFrame
            columns_map = {
                'rolling_3m_return': '3 ' + t('three_months').split()[-1],
                'rolling_1y_return': '1 ' + t('one_year'),
                'rolling_3y_return': '3 ' + t('three_years').split()[-1]
            }
            
            # Rename only the columns that exist
            new_columns = {}
            for old_col, new_col in columns_map.items():
                if old_col in rolling_monthly.columns:
                    new_columns[old_col] = new_col
                    
            rolling_monthly = rolling_monthly.rename(columns=new_columns)
            
            # Remove empty columns
            rolling_monthly = rolling_monthly.loc[:, rolling_monthly.columns != '']
            
            # Create comparison plot with larger dimensions and fonts
            fig_comparison = go.Figure()
            
            colors = ['rgba(65, 105, 225, 0.8)', 'rgba(255, 102, 0, 0.8)', 'rgba(0, 153, 0, 0.8)']
            for i, col in enumerate(rolling_monthly.columns):
                fig_comparison.add_trace(
                    go.Scatter(
                        x=rolling_monthly.index,
                        y=rolling_monthly[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=3)  # Increased line width
                    )
                )
            
            # Adjust layout with larger dimensions and fonts
            fig_comparison.update_layout(
                title=t('comparison_title'),
                height=800,  # Increased height
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=24)  # Increased font size
                ),
                title_font=dict(size=24),  # Increased title font size
                margin=dict(l=60, r=60, t=100, b=60)  # Increased margins
            )
            
            # Update axis fonts
            fig_comparison.update_xaxes(
                title_text=t('date'),
                title_font=dict(size=24),
                tickfont=dict(size=24)
            )
            
            fig_comparison.update_yaxes(
                title_text=t('return') + " (%)",
                title_font=dict(size=24),
                tickfont=dict(size=24)
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Download button for comparison chart with higher resolution
            filename_comparison = f"rolling_comparison_{strategy_display_name_for_file}_{period_text}.png"
            get_image_download_link(
                fig_comparison, 
                filename_comparison, 
                "📥 " + t('download_comparison'),
                width=2400, 
                height=1600, 
                scale=3
            )
            
            # Detailed statistics table
            st.markdown(f"<h4>{t('detailed_statistics')}</h4>", unsafe_allow_html=True)
            
            stats_df = pd.DataFrame({
                t('period'): rolling_monthly.columns,
                t('avg_return_pct'): [round(rolling_monthly[col].mean(), 2) for col in rolling_monthly.columns],
                t('std_dev_pct'): [round(rolling_monthly[col].std(), 2) for col in rolling_monthly.columns],
                t('min_pct'): [round(rolling_monthly[col].min(), 2) for col in rolling_monthly.columns],
                t('max_pct'): [round(rolling_monthly[col].max(), 2) for col in rolling_monthly.columns]
            })
            
            st.table(stats_df)
        else:
            st.warning("Not enough data for any rolling returns period to make a comparison.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# DETAILS TAB - Detailed performance metrics
with details_tab:
    # Replace the date info with more compact version
    date_info_html = f"""
    <div class="date-info">
        <strong>Calculation Period:</strong> {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}
    </div>
    """
    st.markdown(date_info_html, unsafe_allow_html=True)
    
    st.markdown("""<div class="metric-container">
        <h3 class="metric-section-header">📊 Detailed Metrics</h3>
    """, unsafe_allow_html=True)
    
    # Add a style for detailed metrics table - vertical compact layout
    st.markdown("""
    <style>
    .vertical-metrics-table {
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 13px;
    }
    .vertical-metrics-table td {
        padding: 4px 10px;
        border-bottom: 1px solid #e1e4e8;
    }
    .vertical-metrics-table tr:last-child td {
        border-bottom: none;
    }
    .metric-name {
        font-weight: 600;
        text-align: left;
    }
    .metric-value {
        text-align: right;
    }
    .positive-value {
        color: #10b981;
        font-weight: 600;
    }
    .negative-value {
        color: #ef4444;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Helper function to add color to values
    def format_value(value, is_percentage=True, reverse=False):
        if isinstance(value, str):
            return value
        
        if value > 0:
            color_class = "negative-value" if reverse else "positive-value"
        elif value < 0:
            color_class = "positive-value" if reverse else "negative-value"
        else:
            color_class = ""
            
        formatted = f"{value:.2f}{'%' if is_percentage else ''}"
        return f'<span class="{color_class}">{formatted}</span>'
    
    # Create color-coded value strings
    cagr_value = format_value(cagr*100)
    tr_value = format_value(total_return*100)
    vol_value = format_value(annual_volatility*100, reverse=True)
    mdd_value = format_value(max_drawdown, reverse=True)
    sharpe_value = format_value(sharpe, is_percentage=False)
    sortino_value = format_value(sortino, is_percentage=False)
    te_value = format_value(tracking_error*100, reverse=True)
    beta_value = format_value(beta, is_percentage=False)
    alpha_value = format_value(alpha_annualized_pct)
    
    # Create vertical layout HTML for detailed metrics table
    detailed_table_html = f"""
    <table class="vertical-metrics-table">
        <tr>
            <td class="metric-name">CAGR</td>
            <td class="metric-value">{cagr_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Total Return</td>
            <td class="metric-value">{tr_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Volatility (p.a.)</td>
            <td class="metric-value">{vol_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Max Drawdown</td>
            <td class="metric-value">{mdd_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Alpha (p.a.)</td>
            <td class="metric-value">{alpha_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Beta</td>
            <td class="metric-value">{beta_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Sharpe Ratio</td>
            <td class="metric-value">{sharpe_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Sortino Ratio</td>
            <td class="metric-value">{sortino_value}</td>
        </tr>
        <tr>
            <td class="metric-name">Tracking Error</td>
            <td class="metric-value">{te_value}</td>
        </tr>
    </table>
    """
    
    # Display the custom formatted table
    st.markdown(detailed_table_html, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer with credits
st.markdown("""
<div style="text-align: center; margin-top: 20px; padding: 10px; color: #666;">
    <p>© 2024 - Quantmade AI</p>
</div>
""", unsafe_allow_html=True)
