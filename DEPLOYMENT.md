# Deployment Guide for Quantmade AI Dashboard

## Steps to Deploy on Streamlit Cloud

1. **Prepare your GitHub repository**:
   - Create a new GitHub repository (e.g., QuantmadeAIDashboard)
   - Push the contents of the `streamlit_app` directory to this repository

2. **Connect to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Configure the deployment**:
   - Select a branch (usually `main`)
   - Set the Python version (3.9 or higher recommended)
   - The app will be available at a URL like: `https://your-app-name.streamlit.app`

4. **Test the deployment**:
   - Check that the dashboard loads correctly
   - Verify all charts are displayed properly
   - Test the language toggle functionality

5. **Embed in WordPress**:
   - Use an iframe to embed the app in your WordPress site:
   ```html
   <iframe
     src="https://your-app-name.streamlit.app/?embed=true"
     height="800"
     width="100%"
     style="border: none;"
   ></iframe>
   ```

## Troubleshooting

If you encounter the error `ModuleNotFoundError: No module named 'matplotlib'` or similar:

1. Make sure the `requirements.txt` file is in the root of your repository with all necessary dependencies:
   ```
   streamlit==1.31.0
   pandas==2.1.3
   numpy==1.26.3
   matplotlib==3.8.2
   plotly==5.18.0
   pillow==10.1.0
   kaleido==0.2.1
   ```

2. Check the Streamlit Cloud logs by clicking "Manage app" in the lower right corner of your app.

3. Verify all data files are correctly named and located in the expected directories.

4. If changes are needed, update your GitHub repository and Streamlit Cloud will automatically redeploy.

## File Structure Required for Deployment

```
streamlit_app/
├── app.py                 # Main application code
├── requirements.txt       # Required packages
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── data/
    └── PerformancesClean/
        ├── STEADY US 100performance.csvperformance.csv
        └── STEADY US Tech 100performance.csvperformance.csv
```

Remember that all paths in the code should use `os.path.join()` for cross-platform compatibility. 