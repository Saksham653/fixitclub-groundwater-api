"""
Validation utilities for data quality checks and input validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

def validate_dataset(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> Dict[str, Any]:
    """
    Validate basic dataset requirements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    min_rows : int
        Minimum required rows
    min_cols : int
        Minimum required columns
    
    Returns:
    --------
    dict : Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check if dataframe is not None or empty
        if df is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is None")
            return validation_results
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check minimum dimensions
        if len(df) < min_rows:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Dataset has {len(df)} rows, minimum {min_rows} required")
        
        if len(df.columns) < min_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Dataset has {len(df.columns)} columns, minimum {min_cols} required")
        
        # Check for completely missing columns
        completely_missing = []
        for col in df.columns:
            if df[col].isnull().all():
                completely_missing.append(col)
        
        if completely_missing:
            validation_results['warnings'].append(f"Columns with all missing values: {completely_missing}")
        
        # Check data types
        type_info = df.dtypes.value_counts().to_dict()
        validation_results['info']['data_types'] = {str(k): int(v) for k, v in type_info.items()}
        
        # Missing value analysis
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_stats[col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        validation_results['info']['missing_values'] = missing_stats
        
        # High missing value warning
        high_missing = [col for col, stats in missing_stats.items() if stats['percentage'] > 50]
        if high_missing:
            validation_results['warnings'].append(f"Columns with >50% missing values: {high_missing}")
        
        # Duplicate row check
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"{duplicate_count} duplicate rows found")
        
        validation_results['info']['shape'] = df.shape
        validation_results['info']['duplicate_rows'] = int(duplicate_count)
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation error: {str(e)}")
    
    return validation_results

def validate_columns(df: pd.DataFrame, required_columns: List[str], optional_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate required and optional columns in dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    required_columns : list
        List of required column names
    optional_columns : list, optional
        List of optional column names
    
    Returns:
    --------
    dict : Column validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        df_columns = set(df.columns)
        
        # Check required columns
        missing_required = []
        for col in required_columns:
            if col not in df_columns:
                missing_required.append(col)
        
        if missing_required:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check optional columns
        if optional_columns:
            missing_optional = []
            for col in optional_columns:
                if col not in df_columns:
                    missing_optional.append(col)
            
            if missing_optional:
                validation_results['warnings'].append(f"Missing optional columns: {missing_optional}")
        
        # Identify extra columns
        all_expected = set(required_columns)
        if optional_columns:
            all_expected.update(optional_columns)
        
        extra_columns = df_columns - all_expected
        if extra_columns:
            validation_results['info']['extra_columns'] = list(extra_columns)
        
        validation_results['info']['found_columns'] = list(df_columns)
        validation_results['info']['required_found'] = [col for col in required_columns if col in df_columns]
        
        if optional_columns:
            validation_results['info']['optional_found'] = [col for col in optional_columns if col in df_columns]
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Column validation error: {str(e)}")
    
    return validation_results

def validate_numeric_columns(df: pd.DataFrame, columns: List[str], allow_missing: bool = True) -> Dict[str, Any]:
    """
    Validate that specified columns contain numeric data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    columns : list
        List of column names to check
    allow_missing : bool
        Whether to allow missing values
    
    Returns:
    --------
    dict : Numeric validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        for col in columns:
            if col not in df.columns:
                validation_results['errors'].append(f"Column '{col}' not found")
                validation_results['is_valid'] = False
                continue
            
            col_info = {}
            
            # Check data type
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = numeric_series.isnull().sum() - df[col].isnull().sum()
                
                if non_numeric_count > 0:
                    validation_results['warnings'].append(
                        f"Column '{col}' has {non_numeric_count} non-numeric values"
                    )
                    col_info['non_numeric_values'] = int(non_numeric_count)
            
            # Check for missing values
            missing_count = df[col].isnull().sum()
            if missing_count > 0 and not allow_missing:
                validation_results['errors'].append(f"Column '{col}' has missing values")
                validation_results['is_valid'] = False
            
            col_info['missing_count'] = int(missing_count)
            col_info['data_type'] = str(df[col].dtype)
            
            # Basic statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_data = df[col].dropna()
                if len(numeric_data) > 0:
                    col_info['statistics'] = {
                        'min': float(numeric_data.min()),
                        'max': float(numeric_data.max()),
                        'mean': float(numeric_data.mean()),
                        'std': float(numeric_data.std()),
                        'unique_values': int(numeric_data.nunique())
                    }
            
            validation_results['info'][col] = col_info
    
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Numeric validation error: {str(e)}")
    
    return validation_results

def validate_date_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    Validate that specified columns contain date data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    columns : list
        List of column names to check
    
    Returns:
    --------
    dict : Date validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        for col in columns:
            if col not in df.columns:
                validation_results['errors'].append(f"Date column '{col}' not found")
                validation_results['is_valid'] = False
                continue
            
            col_info = {}
            
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info['is_datetime'] = True
                date_series = df[col]
            else:
                # Try to convert to datetime
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    conversion_failures = date_series.isnull().sum() - df[col].isnull().sum()
                    
                    if conversion_failures > 0:
                        validation_results['warnings'].append(
                            f"Column '{col}' has {conversion_failures} values that couldn't be converted to dates"
                        )
                    
                    col_info['is_datetime'] = False
                    col_info['conversion_failures'] = int(conversion_failures)
                    
                except Exception:
                    validation_results['errors'].append(f"Column '{col}' cannot be converted to dates")
                    validation_results['is_valid'] = False
                    continue
            
            # Date range analysis
            valid_dates = date_series.dropna()
            if len(valid_dates) > 0:
                col_info['date_range'] = {
                    'min_date': valid_dates.min().strftime('%Y-%m-%d'),
                    'max_date': valid_dates.max().strftime('%Y-%m-%d'),
                    'total_days': int((valid_dates.max() - valid_dates.min()).days),
                    'unique_dates': int(valid_dates.nunique())
                }
                
                # Check for duplicates
                if valid_dates.nunique() < len(valid_dates):
                    duplicate_count = len(valid_dates) - valid_dates.nunique()
                    validation_results['warnings'].append(
                        f"Column '{col}' has {duplicate_count} duplicate dates"
                    )
                    col_info['duplicate_dates'] = int(duplicate_count)
            
            validation_results['info'][col] = col_info
    
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Date validation error: {str(e)}")
    
    return validation_results

def validate_coordinate_columns(df: pd.DataFrame, lat_column: str, lon_column: str) -> Dict[str, Any]:
    """
    Validate latitude and longitude columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    lat_column : str
        Name of latitude column
    lon_column : str
        Name of longitude column
    
    Returns:
    --------
    dict : Coordinate validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check if columns exist
        for col_name, col_type in [(lat_column, 'latitude'), (lon_column, 'longitude')]:
            if col_name not in df.columns:
                validation_results['errors'].append(f"{col_type.title()} column '{col_name}' not found")
                validation_results['is_valid'] = False
        
        if not validation_results['is_valid']:
            return validation_results
        
        # Validate latitude values
        lat_series = pd.to_numeric(df[lat_column], errors='coerce')
        invalid_lat = ((lat_series < -90) | (lat_series > 90)).sum()
        missing_lat = lat_series.isnull().sum() - df[lat_column].isnull().sum()
        
        if invalid_lat > 0:
            validation_results['warnings'].append(f"{invalid_lat} invalid latitude values (outside -90 to 90)")
        
        if missing_lat > 0:
            validation_results['warnings'].append(f"{missing_lat} non-numeric latitude values")
        
        # Validate longitude values
        lon_series = pd.to_numeric(df[lon_column], errors='coerce')
        invalid_lon = ((lon_series < -180) | (lon_series > 180)).sum()
        missing_lon = lon_series.isnull().sum() - df[lon_column].isnull().sum()
        
        if invalid_lon > 0:
            validation_results['warnings'].append(f"{invalid_lon} invalid longitude values (outside -180 to 180)")
        
        if missing_lon > 0:
            validation_results['warnings'].append(f"{missing_lon} non-numeric longitude values")
        
        # Valid coordinate pairs
        valid_coords = lat_series.notna() & lon_series.notna() & \
                      (lat_series >= -90) & (lat_series <= 90) & \
                      (lon_series >= -180) & (lon_series <= 180)
        
        valid_count = valid_coords.sum()
        
        validation_results['info'] = {
            'total_records': len(df),
            'valid_coordinates': int(valid_count),
            'invalid_coordinates': int(len(df) - valid_count),
            'latitude_range': {
                'min': float(lat_series.min()) if lat_series.notna().any() else None,
                'max': float(lat_series.max()) if lat_series.notna().any() else None
            },
            'longitude_range': {
                'min': float(lon_series.min()) if lon_series.notna().any() else None,
                'max': float(lon_series.max()) if lon_series.notna().any() else None
            }
        }
        
        # Check if coordinates seem reasonable (basic sanity check)
        if valid_count > 0:
            # Check if all coordinates are the same (potential issue)
            if lat_series.nunique() == 1 and lon_series.nunique() == 1:
                validation_results['warnings'].append("All coordinates are identical")
            
            # Check for potential coordinate system issues
            lat_range = lat_series.max() - lat_series.min()
            lon_range = lon_series.max() - lon_series.min()
            
            if lat_range < 0.001 and lon_range < 0.001:
                validation_results['warnings'].append("Very small coordinate range - check if data is in correct units")
    
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Coordinate validation error: {str(e)}")
    
    return validation_results
