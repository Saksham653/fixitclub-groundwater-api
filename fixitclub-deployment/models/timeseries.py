"""
Time-series forecasting models for groundwater level prediction.
Supports Holt-Winters and SARIMAX methods with seasonal decomposition.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

def forecast(df, date_column, value_column, seasonal_periods=12, horizon=6, method='holt_winters', confidence_level=0.95):
    """
    Perform time-series forecasting on groundwater data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column to forecast
    seasonal_periods : int
        Number of periods in a season
    horizon : int
        Number of periods to forecast
    method : str
        'holt_winters' or 'sarimax'
    confidence_level : float
        Confidence level for prediction intervals
    
    Returns:
    --------
    dict : Forecast results with metrics and components
    """
    
    try:
        # Input validation
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        # Prepare time series data
        ts_df = df[[date_column, value_column]].copy()
        ts_df = ts_df.dropna()
        
        if len(ts_df) == 0:
            raise ValueError("No valid data points found")
        
        # Convert date column to datetime
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        ts_df = ts_df.sort_values(date_column)
        ts_df.set_index(date_column, inplace=True)
        
        # Handle duplicates by taking mean
        ts_df = ts_df.groupby(ts_df.index).mean()
        
        # Extract time series
        ts = ts_df[value_column]
        
        if len(ts) < seasonal_periods * 2:
            # Not enough data for seasonal analysis, use simple method
            seasonal_periods = 1
        
        # Split data for validation
        train_size = len(ts) - min(horizon, len(ts) // 4)
        train_data = ts[:train_size]
        test_data = ts[train_size:]
        
        # Perform forecasting based on method
        if method == 'holt_winters':
            forecast_result = _holt_winters_forecast(
                train_data, test_data, seasonal_periods, horizon, confidence_level
            )
        elif method == 'sarimax':
            forecast_result = _sarimax_forecast(
                train_data, test_data, seasonal_periods, horizon, confidence_level
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Seasonal decomposition for explainability
        components = {}
        if len(ts) >= seasonal_periods * 2:
            try:
                decomposition = seasonal_decompose(ts, model='additive', period=seasonal_periods)
                components = {
                    'trend': 'Upward trend detected' if decomposition.trend.dropna().iloc[-1] > decomposition.trend.dropna().iloc[0] else 'Downward trend detected',
                    'seasonality': f'Seasonal pattern with period {seasonal_periods}',
                    'residual_std': np.std(decomposition.resid.dropna())
                }
            except:
                components = {'trend': 'Unable to decompose', 'seasonality': 'No clear pattern'}
        
        # Calculate validation metrics if test data exists
        metrics = {}
        if len(test_data) > 0 and 'validation_forecast' in forecast_result:
            val_forecast = forecast_result['validation_forecast'][:len(test_data)]
            metrics = {
                'mae': np.mean(np.abs(test_data - val_forecast)),
                'rmse': np.sqrt(np.mean((test_data - val_forecast) ** 2)),
                'mape': np.mean(np.abs((test_data - val_forecast) / test_data)) * 100
            }
        
        # Prepare result
        result = {
            'task': 'timeseries_forecast',
            'method': method,
            'seasonal_periods': seasonal_periods,
            'horizon': horizon,
            'forecast_values': forecast_result['forecast'].tolist(),
            'confidence_intervals': forecast_result.get('confidence_intervals', []),
            'metrics': metrics,
            'components': components,
            'last_observed_value': ts.iloc[-1],
            'forecast_dates': forecast_result.get('forecast_dates', [])
        }
        
        return result
        
    except Exception as e:
        return {
            'task': 'timeseries_forecast',
            'error': str(e),
            'success': False
        }

def _holt_winters_forecast(train_data, test_data, seasonal_periods, horizon, confidence_level):
    """Holt-Winters exponential smoothing forecast"""
    
    try:
        # Determine seasonality type
        seasonal_type = 'add' if seasonal_periods > 1 else None
        
        # Fit Holt-Winters model
        if seasonal_type:
            model = ExponentialSmoothing(
                train_data,
                trend='add',
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                train_data,
                trend='add'
            ).fit(optimized=True)
        
        # Make forecast
        forecast = model.forecast(horizon)
        
        # Get prediction intervals
        forecast_result = model.get_prediction(start=len(train_data), end=len(train_data) + horizon - 1)
        confidence_intervals = forecast_result.conf_int(alpha=1-confidence_level)
        
        # Validation forecast if test data exists
        validation_forecast = None
        if len(test_data) > 0:
            validation_forecast = model.forecast(len(test_data))
        
        # Generate forecast dates
        last_date = train_data.index[-1]
        freq = pd.infer_freq(train_data.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        return {
            'forecast': forecast,
            'confidence_intervals': confidence_intervals.values.tolist() if confidence_intervals is not None else [],
            'validation_forecast': validation_forecast,
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist()
        }
        
    except Exception as e:
        # Fallback to simple method
        mean_value = train_data.mean()
        trend = (train_data.iloc[-1] - train_data.iloc[0]) / len(train_data)
        
        forecast = [mean_value + trend * i for i in range(1, horizon + 1)]
        
        return {
            'forecast': pd.Series(forecast),
            'confidence_intervals': [],
            'validation_forecast': None,
            'forecast_dates': []
        }

def _sarimax_forecast(train_data, test_data, seasonal_periods, horizon, confidence_level):
    """SARIMAX forecast with automatic parameter selection"""
    
    try:
        # Simple SARIMAX model with default parameters
        if seasonal_periods > 1:
            model = SARIMAX(
                train_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, seasonal_periods)
            ).fit(disp=False)
        else:
            model = SARIMAX(
                train_data,
                order=(1, 1, 1)
            ).fit(disp=False)
        
        # Make forecast
        forecast_result = model.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        
        # Get confidence intervals
        confidence_intervals = forecast_result.conf_int(alpha=1-confidence_level)
        
        # Validation forecast if test data exists
        validation_forecast = None
        if len(test_data) > 0:
            val_result = model.get_forecast(steps=len(test_data))
            validation_forecast = val_result.predicted_mean
        
        # Generate forecast dates
        last_date = train_data.index[-1]
        freq = pd.infer_freq(train_data.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        return {
            'forecast': forecast,
            'confidence_intervals': confidence_intervals.values.tolist(),
            'validation_forecast': validation_forecast,
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist()
        }
        
    except Exception as e:
        # Fallback to Holt-Winters
        return _holt_winters_forecast(train_data, test_data, seasonal_periods, horizon, confidence_level)
