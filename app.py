"""
KSA Retail Sales Forecasting
==============================
Converts the Google Colab notebook into a standalone Python script.
Run:  python app.py
Output: CSV forecasts + PNG charts saved to ./outputs/
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import seaborn as sns

import holidays
from hijri_converter import convert

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data Generation
# ---------------------------------------------------------------------------
def generate_data() -> pd.DataFrame:
    """Synthesise daily KSA retail sales 2022-2024."""
    print("[1/6] Generating synthetic KSA retail data …")

    date_range = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    n = len(date_range)
    df = pd.DataFrame({"date": date_range})
    df.set_index("date", inplace=True)

    # Base trend
    trend = np.linspace(1000, 2000, n) + np.cumsum(np.random.normal(0, 2, n))

    # Weekly seasonality (KSA weekend shift)
    weekday = df.index.weekday
    weekend_mask = (
        ((df.index < "2024-07-01") & (weekday.isin([3, 4]))) |
        ((df.index >= "2024-07-01") & (weekday.isin([4, 5])))
    )
    weekly_effect = np.where(weekend_mask, 350, 0)

    # Yearly seasonality
    day_of_year = df.index.dayofyear
    yearly_effect = 150 * np.sin(2 * np.pi * (day_of_year / 365.25) - np.pi / 2)

    # KSA public holidays
    sa_holidays = holidays.country_holidays("SA", years=range(2022, 2025))

    # Holiday effects
    holiday_effect = np.zeros(n)
    for i, date in enumerate(df.index):
        if date in sa_holidays:
            name = sa_holidays[date]
            if "Eid" in name:
                holiday_effect[i] = 500
                if i >= 3:
                    holiday_effect[i - 3:i] += 200
            elif "Ramadan" in name:
                holiday_effect[i] = 300
            elif "National Day" in name:
                holiday_effect[i] = 400
            else:
                holiday_effect[i] = 250

    # Ramadan month effect
    def get_ramadan_dates(year):
        dates = []
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    hijri = convert.Gregorian(year, month, day).to_hijri()
                    if hijri.month == 9:
                        dates.append(pd.Timestamp(year, month, day))
                except Exception:
                    continue
        return pd.DatetimeIndex(dates)

    df["is_ramadan"] = 0
    for year in range(2022, 2025):
        ramadan_days = get_ramadan_dates(year)
        df.loc[df.index.isin(ramadan_days), "is_ramadan"] = 1
    ramadan_effect = df["is_ramadan"] * 100

    noise = np.random.normal(0, 50 + (trend / 1000) * 20, n)
    df["sales"] = trend + weekly_effect + yearly_effect + holiday_effect + ramadan_effect + noise
    df["sales"] = np.maximum(df["sales"], 0).round(2)

    path = os.path.join(OUTPUT_DIR, "ksa_retail_sales.csv")
    df.to_csv(path)
    print(f"    Saved → {path}  ({len(df)} rows)")
    return df, sa_holidays, get_ramadan_dates


# ---------------------------------------------------------------------------
# 2. EDA / Visualisation
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, sa_holidays) -> None:
    print("[2/6] Running EDA …")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(df.index, df["sales"], linewidth=0.8, alpha=0.8, color="#2E86AB")
    axes[0].set_title("KSA Retail Sales: Daily Trends (2022-2024)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Sales (SAR)")

    ramadan_periods = df[df["is_ramadan"] == 1].index
    for date in ramadan_periods:
        axes[0].axvline(date, color="#F18F01", alpha=0.1, linewidth=0.5)

    df["year"] = df.index.year
    df["month_day"] = df.index.strftime("%m-%d")
    pivot = df.pivot_table(values="sales", index="month_day", columns="year", aggfunc="mean")
    for year, marker in zip([2022, 2023, 2024], ["o", "s", "^"]):
        axes[1].plot(pivot.index[::30], pivot[year][::30], label=str(year),
                     marker=marker, markersize=3)
    axes[1].set_title("Year-over-Year Comparison (monthly samples)", fontsize=12)
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    sa_holidays_ts = pd.to_datetime(list(sa_holidays.keys()))
    holiday_sales = df[df.index.isin(sa_holidays_ts)]["sales"].mean()
    normal_sales = df[~df.index.isin(sa_holidays_ts)]["sales"].mean()
    impact = ((holiday_sales - normal_sales) / normal_sales) * 100
    axes[2].bar(["Normal Days", "Holidays"], [normal_sales, holiday_sales],
                color=["#A23B72", "#F18F01"], alpha=0.8)
    axes[2].set_title(f"Holiday Impact: +{impact:.1f}% uplift", fontsize=12)
    axes[2].set_ylabel("Avg Sales (SAR)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")

    # STL decomposition
    stl = STL(df["sales"], period=7, robust=True)
    result = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal (Weekly)")
    result.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "stl_decomposition.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame, sa_holidays) -> pd.DataFrame:
    print("[3/6] Engineering features …")

    sa_holidays_ts = pd.to_datetime(list(sa_holidays.keys()))
    df_feat = df.copy()

    for lag in [1, 7, 14, 28, 365]:
        df_feat[f"lag_{lag}d"] = df_feat["sales"].shift(lag)

    for window in [7, 14, 30, 90]:
        s = df_feat["sales"].shift(1)
        df_feat[f"roll_mean_{window}d"] = s.rolling(window, min_periods=1).mean()
        df_feat[f"roll_std_{window}d"]  = s.rolling(window, min_periods=1).std()
        df_feat[f"roll_max_{window}d"]  = s.rolling(window, min_periods=1).max()

    df_feat["ema_7d"]  = df_feat["sales"].shift(1).ewm(span=7).mean()
    df_feat["ema_30d"] = df_feat["sales"].shift(1).ewm(span=30).mean()

    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"]     = df_feat.index.month
    df_feat["quarter"]   = df_feat.index.quarter
    df_feat["year"]      = df_feat.index.year
    df_feat["dayofyear"] = df_feat.index.dayofyear
    df_feat["weekofyear"] = df_feat.index.isocalendar().week

    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    df_feat["dow_sin"]   = np.sin(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["dow_cos"]   = np.cos(2 * np.pi * df_feat["dayofweek"] / 7)

    df_feat["is_holiday"] = df_feat.index.isin(sa_holidays_ts).astype(int)
    df_feat["days_to_holiday"] = df_feat.index.map(
        lambda x: min([abs((x - h).days) for h in sa_holidays_ts if h >= x], default=365)
    )

    result = df_feat.dropna()
    print(f"    Features: {result.shape[1]} columns, {len(result)} rows after dropna")
    return result


# ---------------------------------------------------------------------------
# 4. Prophet
# ---------------------------------------------------------------------------
def run_prophet(df: pd.DataFrame, get_ramadan_dates):
    print("[4/6] Fitting Prophet …")

    prophet_df = df.reset_index()[["date", "sales", "is_ramadan"]]
    prophet_df.columns = ["ds", "y", "is_ramadan"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10.0,
    )
    model.add_country_holidays(country_name="SA")
    model.add_seasonality(
        name="ramadan", period=30.5, fourier_order=5, condition_name="is_ramadan"
    )
    model.fit(prophet_df)

    df_cv = cross_validation(
        model,
        initial="730 days",
        period="90 days",
        horizon="30 days",
        parallel="processes",
    )
    df_p = performance_metrics(df_cv)
    print(f"    Prophet MAE: {df_p['mae'].mean():.2f} | RMSE: {df_p['rmse'].mean():.2f}")

    future = model.make_future_dataframe(periods=90)
    future["is_ramadan"] = future["ds"].isin(get_ramadan_dates(2025)).astype(int)
    forecast = model.predict(future)
    return model, forecast


# ---------------------------------------------------------------------------
# 5. SARIMA
# ---------------------------------------------------------------------------
def run_sarima(df: pd.DataFrame):
    print("[5/6] Fitting SARIMA (grid search) …")

    p = d = q = range(0, 2)
    pdq = list(product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in pdq]

    best_aic, best_params = float("inf"), None
    for param in pdq:
        for s_param in seasonal_pdq:
            try:
                mod = SARIMAX(
                    df["sales"],
                    order=param,
                    seasonal_order=s_param,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_params = res.aic, (param, s_param)
            except Exception:
                continue

    print(f"    Best SARIMA: {best_params}  (AIC: {best_aic:.2f})")
    final = SARIMAX(
        df["sales"],
        order=best_params[0],
        seasonal_order=best_params[1],
    ).fit(disp=False)
    return final


# ---------------------------------------------------------------------------
# 6. XGBoost + Ensemble
# ---------------------------------------------------------------------------
def run_xgboost_and_ensemble(
    df_model: pd.DataFrame,
    df: pd.DataFrame,
    forecast_prophet,
    final_sarima,
    sa_holidays,
):
    print("[6/6] XGBoost CV + ensemble forecast …")

    drop_cols = ["sales", "is_ramadan", "month_day", "year"]
    feature_cols = [c for c in df_model.columns if c not in drop_cols]
    X = df_model[feature_cols]
    y = df_model["sales"]

    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = None
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
        }
        xgb_model = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        # Save the exact feature names the model was trained on
        train_feature_names = X_train.columns.tolist()
        preds = xgb_model.predict(dval)
        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"    Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}")

    # Recursive XGBoost forecast
    sa_holidays_ts = pd.to_datetime(list(sa_holidays.keys()))
    future_dates = pd.date_range(start="2025-01-01", periods=90, freq="D")
    recursive_df = df_model.copy()
    xgb_future = []

    for date in future_dates:
        row = pd.DataFrame(index=[date], columns=feature_cols)
        row.loc[date, "dayofweek"]  = date.dayofweek
        row.loc[date, "month"]      = date.month
        row.loc[date, "quarter"]    = date.quarter
        row.loc[date, "dayofyear"]  = date.dayofyear
        row.loc[date, "weekofyear"] = date.isocalendar().week
        row.loc[date, "month_sin"]  = np.sin(2 * np.pi * date.month / 12)
        row.loc[date, "month_cos"]  = np.cos(2 * np.pi * date.month / 12)
        row.loc[date, "dow_sin"]    = np.sin(2 * np.pi * date.dayofweek / 7)
        row.loc[date, "dow_cos"]    = np.cos(2 * np.pi * date.dayofweek / 7)
        row.loc[date, "is_holiday"] = int(date in sa_holidays_ts)
        row.loc[date, "days_to_holiday"] = min(
            [abs((date - h).days) for h in sa_holidays_ts if h >= date], default=365
        )

        hist = recursive_df["sales"].loc[recursive_df.index < date]
        for lag in [1, 7, 14, 28, 365]:
            lag_date = date - timedelta(days=lag)
            row.loc[date, f"lag_{lag}d"] = (
                hist.loc[lag_date] if lag_date in hist.index else hist.mean() if not hist.empty else 0
            )
        for w in [7, 14, 30, 90]:
            if not hist.empty:
                row.loc[date, f"roll_mean_{w}d"] = hist.rolling(w, min_periods=1).mean().iloc[-1]
                row.loc[date, f"roll_std_{w}d"]  = hist.rolling(w, min_periods=1).std().iloc[-1]
                row.loc[date, f"roll_max_{w}d"]  = hist.rolling(w, min_periods=1).max().iloc[-1]
            else:
                row.loc[date, f"roll_mean_{w}d"] = row.loc[date, f"roll_std_{w}d"] = row.loc[date, f"roll_max_{w}d"] = 0
        row.loc[date, "ema_7d"]  = hist.ewm(span=7).mean().iloc[-1] if not hist.empty else 0
        row.loc[date, "ema_30d"] = hist.ewm(span=30).mean().iloc[-1] if not hist.empty else 0

        # Reindex to EXACTLY match training columns (fixes XGBoost feature name mismatch)
        row = row.reindex(columns=train_feature_names, fill_value=0).fillna(0).astype(float)
        pred = xgb_model.predict(xgb.DMatrix(row, feature_names=train_feature_names))[0]
        xgb_future.append(pred)
        recursive_df.loc[date, "sales"] = pred
        for col in feature_cols:
            recursive_df.loc[date, col] = row.loc[date, col]

    prophet_future = forecast_prophet[
        forecast_prophet["ds"].isin(future_dates)
    ]["yhat"].values
    sarima_future = final_sarima.forecast(steps=90)

    comparison = pd.DataFrame({
        "date":     future_dates,
        "prophet":  prophet_future,
        "sarima":   sarima_future.values,
        "xgboost":  xgb_future,
        "ensemble": (prophet_future + sarima_future.values + np.array(xgb_future)) / 3,
    })

    # Plot ensemble forecast
    plt.figure(figsize=(14, 5))
    for col, color in [("prophet", "#2E86AB"), ("sarima", "#A23B72"),
                       ("xgboost", "#F18F01"), ("ensemble", "#333333")]:
        lw = 2.5 if col == "ensemble" else 1
        plt.plot(comparison["date"], comparison[col], label=col.capitalize(),
                 linewidth=lw, color=color)
    plt.title("KSA Retail Sales Forecast – 2025 Q1 (Ensemble)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Sales (SAR)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "forecast_2025_q1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")

    path = os.path.join(OUTPUT_DIR, "ksa_sales_forecast_2025_q1.csv")
    comparison.to_csv(path, index=False)
    print(f"    Saved → {path}")

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KSA Retail Forecasting Pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plots")
    args = parser.parse_args()

    np.random.seed(42)

    df, sa_holidays, get_ramadan_dates = generate_data()

    if not args.skip_eda:
        run_eda(df, sa_holidays)

    df_model = create_features(df, sa_holidays)

    _, forecast_prophet = run_prophet(df, get_ramadan_dates)
    final_sarima = run_sarima(df)

    run_xgboost_and_ensemble(
        df_model, df, forecast_prophet, final_sarima, sa_holidays
    )

    print(f"\n✅ All done. Outputs saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
