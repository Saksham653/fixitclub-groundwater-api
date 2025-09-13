"""
Example test functions to verify all ML modules work correctly.
"""

import pandas as pd
import numpy as np
from models.tabular import train_and_predict
from models.timeseries import forecast
from models.anomaly import detect
from models.geospatial import map_data
from utils.data_processor import DataProcessor

def run_example_tests():
    """
    Run example tests for all ML modules.
    
    Returns:
    --------
    dict : Test results for each module
    """
    
    results = {}
    processor = DataProcessor()
    
    # Test 1: Tabular prediction
    try:
        df_tabular = processor.create_sample_dataset('tabular', 50)
        result = train_and_predict(df_tabular, target_column='Water_Level_m')
        
        results['Tabular Prediction'] = {
            'success': 'error' not in result,
            'output': result,
            'description': 'RandomForest regression on synthetic groundwater data'
        }
    except Exception as e:
        results['Tabular Prediction'] = {
            'success': False,
            'error': str(e),
            'description': 'RandomForest regression test'
        }
    
    # Test 2: Time-series forecasting
    try:
        df_timeseries = processor.create_sample_dataset('timeseries', 100)
        result = forecast(
            df_timeseries, 
            date_column='Date', 
            value_column='Water_Level_m',
            seasonal_periods=30,
            horizon=10
        )
        
        results['Time-series Forecasting'] = {
            'success': 'error' not in result,
            'output': result,
            'description': 'Holt-Winters forecasting on synthetic time series'
        }
    except Exception as e:
        results['Time-series Forecasting'] = {
            'success': False,
            'error': str(e),
            'description': 'Time-series forecasting test'
        }
    
    # Test 3: Anomaly detection
    try:
        df_anomaly = processor.create_sample_dataset('tabular', 50)
        # Add some artificial anomalies
        anomaly_indices = [5, 15, 25, 35, 45]
        df_anomaly.loc[anomaly_indices, 'Water_Level_m'] = df_anomaly['Water_Level_m'].mean() + 5 * df_anomaly['Water_Level_m'].std()
        
        result = detect(df_anomaly, contamination=0.1, method='isolation_forest')
        
        results['Anomaly Detection'] = {
            'success': 'error' not in result,
            'output': result,
            'description': 'Isolation Forest anomaly detection with artificial outliers'
        }
    except Exception as e:
        results['Anomaly Detection'] = {
            'success': False,
            'error': str(e),
            'description': 'Anomaly detection test'
        }
    
    # Test 4: Geospatial mapping
    try:
        df_geospatial = processor.create_sample_dataset('geospatial', 30)
        result = map_data(
            df_geospatial,
            lat_column='Latitude',
            lon_column='Longitude',
            value_column='Water_Level_m',
            station_id_column='Station_ID'
        )
        
        results['Geospatial Mapping'] = {
            'success': 'error' not in result,
            'output': result,
            'description': 'GeoJSON generation for station mapping'
        }
    except Exception as e:
        results['Geospatial Mapping'] = {
            'success': False,
            'error': str(e),
            'description': 'Geospatial mapping test'
        }
    
    # Test 5: Data processor functionality
    try:
        # Test all three dataset types
        datasets = {}
        for dtype in ['tabular', 'timeseries', 'geospatial']:
            datasets[dtype] = processor.create_sample_dataset(dtype, 20)
            dataset_info = processor.detect_dataset_type(datasets[dtype])
            datasets[f'{dtype}_info'] = dataset_info
        
        results['Data Processor'] = {
            'success': True,
            'output': {
                'created_datasets': list(datasets.keys()),
                'detection_results': {k: v for k, v in datasets.items() if 'info' in k}
            },
            'description': 'Data loading and type detection functionality'
        }
    except Exception as e:
        results['Data Processor'] = {
            'success': False,
            'error': str(e),
            'description': 'Data processor test'
        }
    
    # Test 6: Integration test - Full pipeline
    try:
        # Create a mixed dataset with geospatial and temporal features
        np.random.seed(42)
        n_samples = 30
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        mixed_df = pd.DataFrame({
            'Date': dates,
            'Station_ID': ['ST001'] * n_samples,
            'Latitude': [25.12] * n_samples,
            'Longitude': [81.55] * n_samples,
            'Water_Level_m': np.random.normal(15, 2, n_samples) + 3 * np.sin(2 * np.pi * np.arange(n_samples) / 30),
            'Rainfall_mm': np.random.exponential(5, n_samples),
            'Temperature_C': np.random.normal(28, 4, n_samples)
        })
        
        # Run multiple analyses
        integration_results = {}
        
        # Tabular analysis
        tab_result = train_and_predict(mixed_df, target_column='Water_Level_m')
        integration_results['tabular'] = 'error' not in tab_result
        
        # Time-series analysis
        ts_result = forecast(mixed_df, 'Date', 'Water_Level_m', horizon=5)
        integration_results['timeseries'] = 'error' not in ts_result
        
        # Geospatial analysis
        geo_result = map_data(mixed_df, 'Latitude', 'Longitude', 'Water_Level_m')
        integration_results['geospatial'] = 'error' not in geo_result
        
        # Anomaly detection
        anom_result = detect(mixed_df[['Water_Level_m', 'Rainfall_mm', 'Temperature_C']])
        integration_results['anomaly'] = 'error' not in anom_result
        
        results['Integration Test'] = {
            'success': all(integration_results.values()),
            'output': {
                'analyses_completed': integration_results,
                'dataset_shape': mixed_df.shape,
                'all_passed': all(integration_results.values())
            },
            'description': 'Full pipeline test with mixed dataset'
        }
        
    except Exception as e:
        results['Integration Test'] = {
            'success': False,
            'error': str(e),
            'description': 'Integration test'
        }
    
    return results

def create_test_datasets():
    """
    Create various test datasets for manual testing.
    
    Returns:
    --------
    dict : Dictionary of test datasets
    """
    
    processor = DataProcessor()
    datasets = {}
    
    # Simple tabular dataset
    datasets['simple_tabular'] = pd.DataFrame({
        'Station': ['A', 'B', 'C', 'D', 'E'],
        'Depth_m': [20, 30, 25, 35, 28],
        'Water_Level_m': [12, 18, 15, 22, 16],
        'Soil_Type': ['Sandy', 'Clay', 'Loamy', 'Clay', 'Sandy']
    })
    
    # Time series with missing values
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    values = 10 + 3 * np.sin(2 * np.pi * np.arange(50) / 30) + np.random.normal(0, 0.5, 50)
    # Introduce some missing values
    missing_indices = np.random.choice(50, 5, replace=False)
    values[missing_indices] = np.nan
    
    datasets['timeseries_missing'] = pd.DataFrame({
        'date': dates,
        'water_level': values,
        'station': ['ST001'] * 50
    })
    
    # Geospatial with outliers
    n_stations = 25
    # Most stations clustered around a point
    main_lat = 25.0 + np.random.normal(0, 0.5, 20)
    main_lon = 82.0 + np.random.normal(0, 0.5, 20)
    # Few outlier stations
    outlier_lat = [30.0, 15.0, 28.0, 20.0, 22.0]
    outlier_lon = [75.0, 88.0, 85.0, 77.0, 90.0]
    
    all_lat = np.concatenate([main_lat, outlier_lat])
    all_lon = np.concatenate([main_lon, outlier_lon])
    
    datasets['geospatial_outliers'] = pd.DataFrame({
        'station_id': [f'ST{i:03d}' for i in range(n_stations)],
        'latitude': all_lat,
        'longitude': all_lon,
        'water_level_m': np.random.normal(15, 3, n_stations),
        'aquifer_type': np.random.choice(['Confined', 'Unconfined', 'Perched'], n_stations)
    })
    
    # Mixed dataset for comprehensive testing
    datasets['comprehensive'] = processor.create_sample_dataset('tabular', 100)
    
    return datasets

def run_specific_test(test_name, **kwargs):
    """
    Run a specific test with custom parameters.
    
    Parameters:
    -----------
    test_name : str
        Name of test to run
    **kwargs : dict
        Test-specific parameters
    
    Returns:
    --------
    dict : Test result
    """
    
    try:
        processor = DataProcessor()
        
        if test_name == 'tabular_custom':
            df = kwargs.get('data', processor.create_sample_dataset('tabular'))
            target = kwargs.get('target', 'Water_Level_m')
            task_type = kwargs.get('task_type', 'auto')
            
            result = train_and_predict(df, target_column=target, task_type=task_type)
            return {'success': 'error' not in result, 'output': result}
            
        elif test_name == 'timeseries_custom':
            df = kwargs.get('data', processor.create_sample_dataset('timeseries'))
            date_col = kwargs.get('date_column', 'Date')
            value_col = kwargs.get('value_column', 'Water_Level_m')
            horizon = kwargs.get('horizon', 10)
            method = kwargs.get('method', 'holt_winters')
            
            result = forecast(df, date_col, value_col, horizon=horizon, method=method)
            return {'success': 'error' not in result, 'output': result}
            
        elif test_name == 'anomaly_custom':
            df = kwargs.get('data', processor.create_sample_dataset('tabular'))
            contamination = kwargs.get('contamination', 0.1)
            method = kwargs.get('method', 'isolation_forest')
            
            result = detect(df, contamination=contamination, method=method)
            return {'success': 'error' not in result, 'output': result}
            
        elif test_name == 'geospatial_custom':
            df = kwargs.get('data', processor.create_sample_dataset('geospatial'))
            lat_col = kwargs.get('lat_column', 'Latitude')
            lon_col = kwargs.get('lon_column', 'Longitude')
            value_col = kwargs.get('value_column', 'Water_Level_m')
            
            result = map_data(df, lat_col, lon_col, value_column=value_col)
            return {'success': 'error' not in result, 'output': result}
            
        else:
            return {'success': False, 'error': f'Unknown test: {test_name}'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}
