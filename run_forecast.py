import sys
import numpy as np

# Monkey patch numpy to avoid the float_ issue
sys.modules['numpy'].float_ = np.float64

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from itertools import product
import pickle
import os
import multiprocessing as mp

from functools import partial

import warnings
warnings.filterwarnings("ignore", message="Importing plotly failed. Interactive plots will not work.")

# [CHANGE 1: Added load_covid_data function]
def load_covid_data(file_path):
    print(f"Loading COVID data from {file_path}")
    covid_data = pd.read_csv(file_path)
    covid_data['dt'] = pd.to_datetime(covid_data['date'])
    covid_data = covid_data.rename(columns={'day-wise cases': 'new_cases', 'day-wise deaths': 'new_deaths'})
    print("COVID data loaded.")
    return covid_data[['dt', 'new_cases', 'new_deaths']]

SKU_HYPERPARAMETERS = {
    '10-inch mattresses': {
        'changepoint_prior_scale': 0.2,
        'seasonality_prior_scale': 50.0,
        'holidays_prior_scale': 25.0,
        'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.97,
        'n_changepoints': 55
    },
    '12-inch mattresses': {
        'changepoint_prior_scale': 0.12,
        'seasonality_prior_scale': 40.0,
        'holidays_prior_scale': 25.0,
        'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.92,
        'n_changepoints': 48
    },
    '14-inch mattresses': {
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 30.0,
        'holidays_prior_scale': 18.0,
        'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.88,
        'n_changepoints': 45
    },
    '16-inch mattresses': {
        'changepoint_prior_scale': 0.01,
        'seasonality_prior_scale': 4.5,
        'holidays_prior_scale': 3.0,
        'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.9,
        'n_changepoints': 16
    }
}

def save_checkpoint(state, filename):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def add_lag_features(df):
    print("Adding lag features...")
    df['ordered_quantity'] = df['ordered_quantity'].astype('float64')

    df['lag_1'] = df.groupby('sku')['ordered_quantity'].shift(1)
    df['lag_7'] = df.groupby('sku')['ordered_quantity'].shift(7)
    df['lag_14'] = df.groupby('sku')['ordered_quantity'].shift(14)
    df['lag_30'] = df.groupby('sku')['ordered_quantity'].shift(30)
    df['lag_60'] = df.groupby('sku')['ordered_quantity'].shift(60)
    df['rolling_mean_7'] = df.groupby('sku')['ordered_quantity'].rolling(window=7).mean().reset_index(0, drop=True)
    df['rolling_mean_14'] = df.groupby('sku')['ordered_quantity'].rolling(window=14).mean().reset_index(0, drop=True)
    df['rolling_mean_30'] = df.groupby('sku')['ordered_quantity'].rolling(window=30).mean().reset_index(0, drop=True)

    lag_columns = ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'lag_60', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']
    for col in lag_columns:
        df[col] = df.groupby('sku')[col].transform(lambda x: x.fillna(x.mean()))

    df[lag_columns] = df[lag_columns].fillna(0)
    df[['ordered_quantity'] + lag_columns] = df[['ordered_quantity'] + lag_columns].round().astype('int64')

    print("Lag features added.")
    return df

# [CHANGE 2: Modified add_seasonal_features to include COVID data]
def add_seasonal_features(df, covid_data):
    print("Adding seasonal and COVID features...")
    df['month'] = df['dt'].dt.month
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_summer_peak'] = ((df['month'] == 5) & (df['dt'].dt.day >= 15)) | (df['month'].isin([6, 7])).astype(int)
    df['is_black_friday'] = ((df['month'] == 11) & (df['dt'].dt.day >= 20) & (df['dt'].dt.day <= 30)).astype(int)
    df['is_back_to_school'] = ((df['month'] == 8) & (df['dt'].dt.day >= 15) | (df['month'] == 9) & (df['dt'].dt.day <= 15)).astype(int)
    df['is_holiday_season'] = (df['month'].isin([12, 1])).astype(int)
    df['quarter'] = df['dt'].dt.quarter
    
    # Add COVID features
    df = df.merge(covid_data, on='dt', how='left')
    df['new_cases'] = df['new_cases'].fillna(0)
    df['new_deaths'] = df['new_deaths'].fillna(0)
    df['cases_7day_avg'] = df.groupby('sku')['new_cases'].transform(lambda x: x.rolling(window=7).mean())
    df['deaths_7day_avg'] = df.groupby('sku')['new_deaths'].transform(lambda x: x.rolling(window=7).mean())
    
    print("Seasonal and COVID features added.")
    return df

# [CHANGE 3: Modified forecast_sku to include COVID features]
def forecast_sku(args):
    row, df_cutoff, holiday_df, cutoff, forecast_periods = args
    sku = row['sku']
    print(f"Processing SKU: {sku}")
    product_data = df_cutoff[df_cutoff['sku'] == sku][['dt', 'ordered_quantity', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'lag_60', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30', 'is_weekend', 'is_summer_peak', 'is_black_friday', 'is_back_to_school', 'is_holiday_season', 'quarter', 'cases_7day_avg', 'deaths_7day_avg']]
    product_data = product_data.rename(columns={'dt': 'ds', 'ordered_quantity': 'y'})

    if len(product_data) < 30:
        print(f"Skipping SKU {sku} due to insufficient data")
        return None

    if product_data.isnull().any().any():
        print(f"Warning: NaN values found for SKU {sku}. Filling with 0.")
        product_data = product_data.fillna(0)

    # Use the provided hyperparameters
    best_params = SKU_HYPERPARAMETERS[sku]
    print(f"Using hyperparameters for SKU {sku}: {best_params}")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holiday_df,
        **best_params
    )

    for col in ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'lag_60', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30', 'is_weekend', 'is_summer_peak', 'is_black_friday', 'is_back_to_school', 'is_holiday_season', 'quarter', 'cases_7day_avg', 'deaths_7day_avg']:
        model.add_regressor(col)

    model.fit(product_data)
    print(f"Model fitted for SKU {row['sku']}")

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods)
    
    # Fill future values for lag features
    for col in ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'lag_60', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']:
        future[col] = product_data[col].rolling(window=3, min_periods=1).mean().iloc[-1]

    # Add seasonal features to future dataframe
    future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    future['is_summer_peak'] = ((future['ds'].dt.month == 5) & (future['ds'].dt.day >= 15)) | (future['ds'].dt.month.isin([6, 7])).astype(int)
    future['is_black_friday'] = ((future['ds'].dt.month == 11) & (future['ds'].dt.day >= 20) & (future['ds'].dt.day <= 30)).astype(int)
    future['is_back_to_school'] = ((future['ds'].dt.month == 8) & (future['ds'].dt.day >= 15) | (future['ds'].dt.month == 9) & (future['ds'].dt.day <= 15)).astype(int)
    future['is_holiday_season'] = (future['ds'].dt.month.isin([12, 1])).astype(int)
    future['quarter'] = future['ds'].dt.quarter

    # Add COVID features to future dataframe
    future['cases_7day_avg'] = product_data['cases_7day_avg'].rolling(window=3, min_periods=1).mean().iloc[-1]
    future['deaths_7day_avg'] = product_data['deaths_7day_avg'].rolling(window=3, min_periods=1).mean().iloc[-1]

    forecast = model.predict(future)
    print(f"Predictions made for SKU {row['sku']}")

    forecast['month'] = forecast['ds'].dt.month
    forecast['year'] = forecast['ds'].dt.year
    forecast['data_cutoff_date'] = cutoff

    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['data_cutoff_date'] = pd.to_datetime(forecast['data_cutoff_date'])

    forecast['month_diff'] = forecast.apply(lambda row:
                                            (row['ds'].year - row['data_cutoff_date'].year) * 12 +
                                            row['ds'].month - row['data_cutoff_date'].month, axis=1)

    forecast_month_aggregate = forecast.groupby(['month','year','month_diff']).agg(
        sales=('yhat', 'sum')
    ).reset_index()
    forecast_month_aggregate['sku'] = row['sku']

    return forecast_month_aggregate

# [CHANGE 4: Modified run_forecast_model to include COVID data]
def run_forecast_model(df, covid_data, checkpoint_file='forecast_checkpoint.pkl'):
    print("Starting forecast model...")

    # Determine if a fresh start is requested
    if start_fresh:
        checkpoint_file = 'forecast_checkpoint_new.pkl'
        print("Starting a fresh forecast...")
        # Clear the checkpoint file if it exists
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(checkpoint_file + '.backup'):
            os.remove(checkpoint_file + '.backup')
        start_cutoff_index = 0
        final_predictions = pd.DataFrame()
        current_sku = None
        backup_checkpoint_file = checkpoint_file + '.backup'  # Initialize this for fresh start as well
    else:
        print("Attempting to resume from existing checkpoint...")
        checkpoint = load_checkpoint(checkpoint_file)
        backup_checkpoint_file = checkpoint_file + '.backup'

        if checkpoint and 'version' in checkpoint and checkpoint['version'] == '2.0':
            print("Checkpoint loaded. Resuming from previous state.")
            final_predictions = checkpoint['final_predictions']
            start_cutoff_index = checkpoint['cutoff_index']
            df = checkpoint['df']
            current_sku = checkpoint.get('current_sku', None)
        elif os.path.exists(backup_checkpoint_file):
            print("Main checkpoint not found or incompatible. Attempting to load backup checkpoint...")
            backup_checkpoint = load_checkpoint(backup_checkpoint_file)
            if backup_checkpoint and 'version' in backup_checkpoint and backup_checkpoint['version'] == '2.0':
                print("Backup checkpoint loaded. Resuming from previous state.")
                final_predictions = backup_checkpoint['final_predictions']
                start_cutoff_index = backup_checkpoint['cutoff_index']
                df = backup_checkpoint['df']
                current_sku = backup_checkpoint.get('current_sku', None)
            else:
                print("No compatible checkpoints found. Starting from the beginning.")
                start_cutoff_index = 0
                final_predictions = pd.DataFrame()
                current_sku = None
        else:
            print("No compatible checkpoints found. Starting from the beginning.")
            start_cutoff_index = 0
            final_predictions = pd.DataFrame()
            current_sku = None

    if start_cutoff_index == 0:
        df['dt'] = pd.to_datetime(df['dt'])
        df.sort_values(by=["sku", "dt"], inplace=True)
        df = add_lag_features(df)
        df = add_seasonal_features(df, covid_data)  # Pass covid_data here

    print(f"Data loaded. Shape: {df.shape}")

    # Find the last complete month in the data
    last_complete_month = df['dt'].max().replace(day=1) - pd.DateOffset(days=1)
    print(f"Last complete month: {last_complete_month.strftime('%Y-%m-%d')}")

    # Filter data up to the last complete month
    df = df[df['dt'] <= last_complete_month]

    # Calculate forecast periods (3 months from the last complete month)
    forecast_end_date = last_complete_month + pd.DateOffset(months=3)
    forecast_periods = (forecast_end_date - last_complete_month).days

    cutoff_dates = [last_complete_month]

    for cutoff_index, cutoff in enumerate(cutoff_dates[start_cutoff_index:], start=start_cutoff_index):
        print(f"Processing cutoff date: {cutoff}")
        df_cutoff = df[df['dt'] <= cutoff]
        unique_products = df_cutoff[['sku']].drop_duplicates()

        cal = calendar()
        holidays = cal.holidays(start=df_cutoff.dt.min(), end=df_cutoff.dt.max(), return_name=True)
        holiday_df = pd.DataFrame(data=holidays, columns=['holiday']).reset_index().rename(columns={'index': 'ds'})

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(forecast_sku, [(row, df_cutoff, holiday_df, cutoff, forecast_periods) 
                                              for _, row in unique_products.iterrows() 
                                              if current_sku is None or row['sku'] > current_sku])

        for result in results:
            if result is not None:
                if final_predictions.empty:
                    final_predictions = result.copy()
                else:
                    final_predictions = pd.concat([final_predictions, result], ignore_index=True)

        # Save checkpoint after each cutoff date
        checkpoint = {
            'version': '2.0',
            'final_predictions': final_predictions,
            'cutoff_index': cutoff_index + 1,
            'df': df
        }
        save_checkpoint(checkpoint, checkpoint_file)
        save_checkpoint(checkpoint, backup_checkpoint_file)
        print(f"Checkpoint saved after cutoff date: {cutoff}")

    df['month'] = df['dt'].dt.month
    df['year'] = df['dt'].dt.year
    summary_df = df.groupby(['year', 'month', 'sku']).agg(
        Actual = ('ordered_quantity', 'sum')
    ).reset_index()

    final_output = summary_df.copy()

    # Verify that 'month_diff' exists
    if 'month_diff' not in final_predictions.columns:
        print("Error: 'month_diff' column is missing from final_predictions.")
        raise KeyError("'month_diff' column is missing from final_predictions.")

    # If 'month_diff' exists, proceed with aggregation
    forecast_3 = final_predictions[final_predictions['month_diff'].isin([1, 2, 3])].copy()
    forecast_3 = forecast_3.groupby(['year', 'month', 'sku']).agg({'sales': 'mean'}).reset_index()
    forecast_3 = forecast_3.rename(columns={'sales': 'Three_Month_Forecast'})

    final_output = final_output.merge(forecast_3, on=['year', 'month', 'sku'], how='outer')

    final_output = final_output.sort_values(['sku', 'year', 'month'])

    # Ensure all forecasted dates are included
    final_output['date'] = pd.to_datetime(final_output['year'].astype(str) + '-' + final_output['month'].astype(str) + '-01')
    final_output = final_output[final_output['date'] <= forecast_end_date]

    final_output['Three_Month_Forecast'] = final_output['Three_Month_Forecast'].round().astype('Int64')

    final_output['Three_Month_Forecast_Diff%'] = ((final_output['Three_Month_Forecast'] - final_output['Actual']) / final_output['Actual']) * 100

    print("Forecast model completed.")
    return final_output

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def plot_forecast(final_output):
    print("Plotting forecasts...")
    
    # Create 'plots' directory if it doesn't exist
    plots_dir = os.path.join(os.getcwd(), 'Forecast Plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved in: {plots_dir}")
    
    plt.style.use('seaborn-v0_8')

    current_year = datetime.now().year
    start_date = pd.to_datetime(f'{current_year}-01-01')
    end_date = final_output['date'].max()

    for sku in final_output['sku'].unique():
        print(f"Processing plot for SKU: {sku}")
        sku_data = final_output[(final_output['sku'] == sku) & (final_output['date'] >= start_date) & (final_output['date'] <= end_date)].sort_values('date')

        if sku_data.empty:
            print(f"No data available for SKU: {sku} in the specified date range, skipping plot.")
            continue

        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot actual data
        actual_data = sku_data[sku_data['Actual'].notna()]
        ax.plot(actual_data['date'], actual_data['Actual'], label='Actual', color='black', linewidth=2, marker='o')
        ax.fill_between(actual_data['date'],
                        actual_data['Actual'] * 0.9,
                        actual_data['Actual'] * 1.1,
                        alpha=0.2,
                        color='gray',
                        label='±10% Actual')

        # Plot forecast data
        forecast_data = sku_data[sku_data['Three_Month_Forecast'].notna()]
        if not forecast_data.empty:
            ax.plot(forecast_data['date'], forecast_data['Three_Month_Forecast'],
                    label='3-Month Forecast',
                    color='#1f77b4',
                    linewidth=1.5,
                    marker='o',
                    markersize=4)

        ax.set_title(f'Actual vs 3-Month Forecast for {sku} ({current_year})', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Mattresses', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_xlim(start_date, end_date)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        filename = f'forecast_{sku.replace(" ", "_")}_{current_year}.png'
        filepath = os.path.join(plots_dir, filename)
        print(f"Saving plot: {filepath}")
        try:
            plt.savefig(filepath)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot for {sku}: {str(e)}")

    print(f"All forecast plots for {current_year} have been saved.")
    print("Saved plot files:")
    for file in os.listdir(plots_dir):
        if file.endswith(f"_{current_year}.png"):
            print(file)

# [CHANGE 5: Modified main script to load and use COVID data]
if __name__ == "__main__":
    try:
        print("Starting the forecasting process...")

        # Load your sales data
        print("Loading historical data...")
        df = pd.read_csv('Sales Historical Data - 2nd May 2025.csv')
        print(f"Historical data loaded. Shape: {df.shape}")

        # Load COVID data
        covid_data = load_covid_data('Forecasting for Mattresses - Covid Data.csv')

        # Ask user if they want to start fresh
        start_fresh = input("Do you want to start a fresh forecast ignoring any existing checkpoints? (y/n): ").lower() == 'y'

        if start_fresh:
            checkpoint_file = 'forecast_checkpoint_new.pkl'
            print("Starting a fresh forecast...")
        else:
            checkpoint_file = 'forecast_checkpoint.pkl'
            print("Attempting to resume from existing checkpoint...")

        # Run the forecast model with checkpointing
        print("Running forecast model...")
        final_output = run_forecast_model(df, covid_data, checkpoint_file=checkpoint_file)

        # Plot the forecast
        print("Plotting forecasts...")
        plot_forecast(final_output)

        # Save the final output to a CSV file
        print("Saving forecast results...")
        final_output.to_csv('forecast_results.csv', index=False)
        print("Forecast results saved to 'forecast_results.csv'")

        print("Forecasting process completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    input("Press Enter to exit...")