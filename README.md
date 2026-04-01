# KSA Retail Sales Forecasting

Time series forecasting model for Saudi Arabian retail sales, incorporating Islamic calendar effects and local holidays.

Files included
- ksa_retail_sales_.csv: Synthetic daily sales data (2022-2024) with KSA-specific patterns
- ksa_sales_forecast_.csv: 90-day forecast for 2025 Q1
- requirements.txt: Python dependencies
- notebook.ipynb: Main analysis notebook (download separately from Colab)

Quick start
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook in Google Colab or Jupyter
3. Run all cells to reproduce the analysis

Model highlights
- Prophet 1.3.0 with KSA holidays
- XGBoost with engineered features
- SARIMA auto-selection
- Ensemble forecast

For more details, see the original Colab notebook.
