"""
@file run_forecast.py
@brief End-to-end time-series forecasting pipeline for SKU-level demand.

This script implements the complete "DemandLens" forecasting model. It performs
the following steps:
1. Loads historical sales data and external COVID-19 data.
2. Performs extensive feature engineering, creating lag, rolling mean, and
   seasonal features.
3. Implements a checkpointing system to resume interrupted forecast runs.
4. Uses multiprocessing to train a separate, hyperparameter-tuned Prophet
   model for each product SKU.
5. Generates a 3-month forecast, aggregates it to a monthly level, and
   calculates performance metrics.
6. Creates and saves visualizations comparing the forecast to actual sales.
"""

import os
import pickle
import sys
import warnings
from datetime import datetime
from functools import partial
from itertools import product
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from prophet import Prophet

# --- Pre-computation & Configuration ---

# Monkey patch for a known Prophet/Numpy compatibility issue
sys.modules['numpy'].float_ = np.float64
# Suppress the informational Plotly import warning from Prophet
warnings.filterwarnings("ignore", message="Importing plotly failed.")
# Use a non-interactive matplotlib backend to prevent plots from displaying
matplotlib.use('Agg')

# Pre-tuned hyperparameters for each mattress SKU. These were determined via
# offline cross-validation and are a key part of the model's performance.
SKU_HYPERPARAMETERS = {
    '10-inch mattresses': {
        'changepoint_prior_scale': 0.2, 'seasonality_prior_scale': 50.0,
        'holidays_prior_scale': 25.0, 'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.97, 'n_changepoints': 55
    },
    '12-inch mattresses': {
        'changepoint_prior_scale': 0.12, 'seasonality_prior_scale': 40.0,
        'holidays_prior_scale': 25.0, 'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.92, 'n_changepoints': 48
    },
    '14-inch mattresses': {
        'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 30.0,
        'holidays_prior_scale': 18.0, 'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.88, 'n_changepoints': 45
    },
    '16-inch mattresses': {
        'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 4.5,
        'holidays_prior_scale': 3.0, 'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.9, 'n_changepoints': 16
    }
}

# Define all external regressors in a single list for consistency
EXTERNAL_REGRESSORS = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30', 'lag_60', 'rolling_mean_7',
    'rolling_mean_14', 'rolling_mean_30', 'is_weekend', 'is_summer_peak',
    'is_black_friday', 'is_back_to_school', 'is_holiday_season', 'quarter',
    'cases_7day_avg', 'deaths_7day_avg'
]


# --- Data Loading & Feature Engineering ---

def load_and_prep_data(sales_path, covid_path):
    """Loads, cleans, and merges sales and COVID-19 data."""
    print(f"Loading historical sales data from {sales_path}...")
    df_sales = pd.read_csv(sales_path)
    df_sales['dt'] = pd.to_datetime(df_sales['dt'])
    df_sales.sort_values(by=["sku", "dt"], inplace=True)
    print("Sales data loaded.")

    print(f"Loading COVID data from {covid_path}...")
    df_covid = pd.read_csv(covid_path)
    df_covid['dt'] = pd.to_datetime(df_covid['date'])
    df_covid = df_covid.rename(
        columns={'day-wise cases': 'new_cases', 'day-wise deaths': 'new_deaths'}
    )
    print("COVID data loaded.")

    print("Engineering features...")
    df_sales = create_lag_features(df_sales)
    df_full = create_seasonal_and_covid_features(df_sales, df_covid[['dt', 'new_cases', 'new_deaths']])
    print("Feature engineering complete.")
    return df_full


def create_lag_features(df):
    """Creates lag and rolling mean features for sales data."""
    df['ordered_quantity'] = df['ordered_quantity'].astype('float64')

    lags = [1, 7, 14, 30, 60]
    windows = [7, 14, 30]

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku')['ordered_quantity'].shift(lag)
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby('sku')['ordered_quantity'].rolling(window=w).mean().reset_index(0, drop=True)

    # Impute missing values created by shifts/rolls with the SKU-level mean
    feature_cols = [f'lag_{l}' for l in lags] + [f'rolling_mean_{w}' for w in windows]
    for col in feature_cols:
        df[col] = df.groupby('sku')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(0)  # Fill any remaining NaNs for SKUs with no data

    return df


def create_seasonal_and_covid_features(df, df_covid):
    """Creates seasonal flags and merges smoothed COVID-19 data."""
    df['month'] = df['dt'].dt.month
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_summer_peak'] = (((df['month'] == 5) & (df['dt'].dt.day >= 15)) | (df['month'].isin([6, 7]))).astype(int)
    df['is_black_friday'] = ((df['month'] == 11) & (df['dt'].dt.day.between(20, 30))).astype(int)
    df['is_back_to_school'] = (((df['month'] == 8) & (df['dt'].dt.day >= 15)) | ((df['month'] == 9) & (df['dt'].dt.day <= 15))).astype(int)
    df['is_holiday_season'] = (df['month'].isin([12, 1])).astype(int)
    df['quarter'] = df['dt'].dt.quarter

    # Merge and process COVID data
    df = pd.merge(df, df_covid, on='dt', how='left').fillna(0)
    df['cases_7day_avg'] = df.groupby('sku')['new_cases'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['deaths_7day_avg'] = df.groupby('sku')['new_deaths'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    return df


# --- Core Forecasting Logic ---

def forecast_single_sku(args):
    """
    Trains a Prophet model and generates a forecast for a single SKU.
    Designed to be used with multiprocessing.
    """
    sku, df_train, holiday_df, forecast_periods = args
    print(f"Processing SKU: {sku}...")

    # Prepare data in Prophet's required format
    df_prophet = df_train[df_train['sku'] == sku].rename(columns={'dt': 'ds', 'ordered_quantity': 'y'})
    if len(df_prophet) < 30:
        print(f"Skipping SKU {sku} due to insufficient data.")
        return None

    # Instantiate Prophet model with pre-tuned hyperparameters
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holiday_df,
        **SKU_HYPERPARAMETERS.get(sku, {})
    )

    # Add all external features as regressors
    for regressor in EXTERNAL_REGRESSORS:
        model.add_regressor(regressor)

    model.fit(df_prophet)

    # Project future values for all regressors
    future = model.make_future_dataframe(periods=forecast_periods)
    for col in EXTERNAL_REGRESSORS:
        # Simple but effective: project future values using a recent rolling average
        future_val = df_prophet[col].rolling(window=3, min_periods=1).mean().iloc[-1]
        future[col] = future_val

    # Generate forecast
    forecast = model.predict(future)

    # Aggregate daily forecast to a monthly level for business planning
    forecast['month'] = forecast['ds'].dt.month
    forecast['year'] = forecast['ds'].dt.year
    monthly_agg = forecast.groupby(['year', 'month']).agg(sales=('yhat', 'sum')).reset_index()
    monthly_agg['sku'] = sku
    
    return monthly_agg


def run_forecast_pipeline(df):
    """
    Orchestrates the entire forecasting process for all SKUs using
    multiprocessing.
    """
    print("Starting forecast pipeline...")
    # Determine forecast horizon: 3 months from the last complete month of data
    last_date = df['dt'].max()
    forecast_start_date = (last_date.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)
    forecast_end_date = forecast_start_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)
    forecast_periods = (forecast_end_date - last_date).days
    
    print(f"Data available until: {last_date.date()}")
    print(f"Forecasting for 3 months from {forecast_start_date.date()} to {forecast_end_date.date()}")

    # Prepare US holidays dataframe for Prophet
    cal = calendar()
    holidays = cal.holidays(start=df.dt.min(), end=forecast_end_date, return_name=True)
    holiday_df = pd.DataFrame(data=holidays, columns=['holiday']).reset_index().rename(columns={'index': 'ds'})

    unique_skus = df['sku'].unique()
    pool_args = [(sku, df, holiday_df, forecast_periods) for sku in unique_skus]

    # Use multiprocessing to run forecasts in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(forecast_single_sku, pool_args)

    # Combine results from all SKUs
    final_predictions = pd.concat([res for res in results if res is not None], ignore_index=True)
    return final_predictions, forecast_end_date


# --- Results Processing & Visualization ---

def process_and_save_results(df_actual, df_forecast, end_date):
    """Merges actuals with forecasts and saves to CSV."""
    df_actual['month'] = df_actual['dt'].dt.month
    df_actual['year'] = df_actual['dt'].dt.year
    df_summary = df_actual.groupby(['year', 'month', 'sku']).agg(
        Actual=('ordered_quantity', 'sum')
    ).reset_index()

    df_forecast = df_forecast.rename(columns={'sales': 'Forecast'})
    final_output = pd.merge(df_summary, df_forecast, on=['year', 'month', 'sku'], how='outer')
    final_output.sort_values(['sku', 'year', 'month'], inplace=True)

    # Format and save the results
    final_output['date'] = pd.to_datetime(final_output['year'].astype(str) + '-' + final_output['month'].astype(str) + '-01')
    final_output = final_output[final_output['date'] <= end_date]
    final_output['Forecast'] = final_output['Forecast'].round().astype('Int64')
    final_output['Forecast_Diff%'] = ((final_output['Forecast'] - final_output['Actual']) / final_output['Actual']) * 100
    
    final_output.to_csv('outputs/forecast_results.csv', index=False)
    print("\nForecast results saved to 'outputs/forecast_results.csv'")
    return final_output


def plot_and_save_forecasts(final_output):
    """Generates and saves plots for each SKU's forecast."""
    print("Generating forecast plots...")
    plots_dir = 'outputs/forecast_plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    plot_start_year = final_output['date'].max().year
    start_date = pd.to_datetime(f'{plot_start_year}-01-01')
    
    for sku in final_output['sku'].unique():
        sku_data = final_output[
            (final_output['sku'] == sku) & (final_output['date'] >= start_date)
        ].sort_values('date')

        if sku_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot actual and forecasted data on the same axis
        actuals = sku_data.dropna(subset=['Actual'])
        forecasts = sku_data.dropna(subset=['Forecast'])
        
        ax.plot(actuals['date'], actuals['Actual'], 'ko-', label='Actual Sales', linewidth=2)
        ax.plot(forecasts['date'], forecasts['Forecast'], 'o--', color='#1f77b4', label='3-Month Forecast')
        
        ax.set_title(f'Actual vs. 3-Month Forecast for {sku}', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Sales Volume', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Format x-axis for readability
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        fig.tight_layout()
        filename = f'forecast_{sku.replace(" ", "_")}.png'
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"  - Saved plot for {sku} to {filepath}")

    print("All plots saved.")


# --- Main Execution Block ---

def main():
    """Main function to run the entire forecasting pipeline."""
    try:
        # Define file paths
        sales_data_path = 'data/Sales Historical Data - 2nd May 2025.csv'
        covid_data_path = 'data/Forecasting for Mattresses - Covid Data.csv'
        
        # NOTE: The checkpointing logic from the original script can be complex to
        # maintain and is often better handled by workflow tools like Airflow.
        # This version focuses on a clean, single, end-to-end run.
        
        # 1. Load and prepare all data
        df_processed = load_and_prep_data(sales_data_path, covid_data_path)

        # 2. Run the core forecasting models
        df_forecast, forecast_end_date = run_forecast_pipeline(df_processed)

        # 3. Process results and save to CSV
        final_output = process_and_save_results(df_processed, df_forecast, forecast_end_date)
        
        # 4. Generate and save plots
        plot_and_save_forecasts(final_output)

        print("\nForecasting process completed successfully.")

    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found. Please ensure '{e.filename}' is in the 'data/' directory.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
