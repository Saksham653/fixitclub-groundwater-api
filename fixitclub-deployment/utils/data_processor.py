"""
Data processing utilities for groundwater analysis.
Handles file loading, preprocessing, and dataset type detection.
"""

import pandas as pd
import numpy as np
from io import StringIO
import re
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Utility class for data processing operations"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
    
    def load_data(self, file_object):
        """
        Load data from uploaded file object.
        
        Parameters:
        -----------
        file_object : streamlit.UploadedFile
            Uploaded file object from Streamlit
        
        Returns:
        --------
        pandas.DataFrame : Loaded dataset
        """
        
        try:
            # Get file extension
            file_name = file_object.name.lower()
            file_extension = file_name.split('.')[-1]
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Load based on file type
            if file_extension == 'csv':
                # Try to detect delimiter
                sample = file_object.read(1024).decode('utf-8', errors='ignore')
                file_object.seek(0)
                
                delimiter = self._detect_delimiter(sample)
                df = pd.read_csv(file_object, delimiter=delimiter)
                
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_object)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic preprocessing
            df = self._preprocess_dataframe(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def _detect_delimiter(self, sample_text):
        """Detect CSV delimiter from sample text"""
        
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            count = sample_text.count(delimiter)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)
        else:
            return ','  # Default to comma
    
    def _preprocess_dataframe(self, df):
        """Basic preprocessing of loaded dataframe"""
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)  # Remove empty rows
        df = df.dropna(how='all', axis=1)  # Remove empty columns
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If more than half can be converted, make it numeric
                    valid_ratio = numeric_series.notna().sum() / len(df)
                    if valid_ratio > 0.5:
                        df[col] = numeric_series
        
        # Try to detect and convert date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                if self._is_date_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass  # Keep as object if conversion fails
        
        return df
    
    def _is_date_column(self, series):
        """Check if a series contains date-like strings"""
        
        # Sample a few non-null values
        sample_values = series.dropna().head(10).astype(str)
        
        if len(sample_values) == 0:
            return False
        
        date_matches = 0
        for value in sample_values:
            for pattern in self.date_patterns:
                if re.search(pattern, value):
                    date_matches += 1
                    break
        
        # If more than half look like dates, consider it a date column
        return date_matches / len(sample_values) > 0.5
    
    def detect_dataset_type(self, df):
        """
        Automatically detect dataset type and key features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        
        Returns:
        --------
        dict : Dataset type and detected features
        """
        
        features = {}
        dataset_type = 'tabular'  # Default
        
        # Check for temporal features
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or self._is_date_column(df[col]):
                date_columns.append(col)
        
        if date_columns:
            features['temporal'] = f"Date columns found: {', '.join(date_columns)}"
            if len(date_columns) >= 1 and len(df) > 10:
                dataset_type = 'timeseries'
        
        # Check for geospatial features
        lat_columns = [col for col in df.columns if 
                      any(keyword in col.lower() for keyword in ['lat', 'latitude'])]
        lon_columns = [col for col in df.columns if 
                      any(keyword in col.lower() for keyword in ['lon', 'longitude', 'long'])]
        
        if lat_columns and lon_columns:
            features['geospatial'] = f"Coordinates found: {lat_columns[0]}, {lon_columns[0]}"
            if dataset_type == 'tabular':  # Don't override timeseries
                dataset_type = 'geospatial'
        
        # Check for ID columns
        id_columns = [col for col in df.columns if 
                     any(keyword in col.lower() for keyword in ['id', 'station', 'site', 'well'])]
        if id_columns:
            features['identifiers'] = f"ID columns: {', '.join(id_columns)}"
        
        # Analyze data quality
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        features['data_quality'] = f"{missing_percentage:.1f}% missing values"
        
        # Detect potential target columns for supervised learning
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            # Look for common groundwater target patterns
            target_patterns = ['level', 'depth', 'height', 'water', 'gwl', 'wl']
            potential_targets = []
            
            for col in numeric_columns:
                if any(pattern in col.lower() for pattern in target_patterns):
                    potential_targets.append(col)
            
            if potential_targets:
                features['potential_targets'] = f"Possible targets: {', '.join(potential_targets)}"
            else:
                features['potential_targets'] = f"Numeric columns: {', '.join(numeric_columns[:3])}"
        
        return {
            'type': dataset_type,
            'features': features,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        }
    
    def handle_missing_values(self, df, strategy='auto', threshold=0.5):
        """
        Handle missing values in dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        strategy : str
            'auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        threshold : float
            Threshold for dropping columns/rows with too many missing values
        
        Returns:
        --------
        pandas.DataFrame : Dataset with handled missing values
        """
        
        df_clean = df.copy()
        
        # Drop columns with too many missing values
        missing_ratio_cols = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_ratio_cols[missing_ratio_cols > threshold].index.tolist()
        df_clean = df_clean.drop(columns=cols_to_drop)
        
        if strategy == 'auto':
            # Automatic strategy based on column type and missing percentage
            for col in df_clean.columns:
                missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                
                if missing_pct == 0:
                    continue
                elif missing_pct > threshold:
                    # Already dropped above
                    continue
                elif df_clean[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col].fillna(mode_val.iloc[0], inplace=True)
                elif df_clean[col].dtype in ['datetime64[ns]']:
                    # Forward fill for dates
                    df_clean[col].fillna(method='ffill', inplace=True)
                else:
                    # Fill numeric with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
        
        elif strategy == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
        elif strategy == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        elif strategy == 'fill_mode':
            for col in df_clean.columns:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col].fillna(mode_val.iloc[0], inplace=True)
        
        return df_clean
    
    def encode_categorical_features(self, df, encoding_type='onehot', max_categories=20):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        encoding_type : str
            'onehot' or 'label'
        max_categories : int
            Maximum number of categories for one-hot encoding
        
        Returns:
        --------
        pandas.DataFrame : Dataset with encoded features
        """
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if encoding_type == 'onehot' and unique_values <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            else:
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def create_sample_dataset(self, dataset_type='tabular', n_samples=100):
        """
        Create sample datasets for testing.
        
        Parameters:
        -----------
        dataset_type : str
            'tabular', 'timeseries', or 'geospatial'
        n_samples : int
            Number of samples to generate
        
        Returns:
        --------
        pandas.DataFrame : Sample dataset
        """
        
        np.random.seed(42)
        
        if dataset_type == 'tabular':
            data = {
                'Station_ID': [f'ST{i:03d}' for i in range(n_samples)],
                'Rainfall_mm': np.random.normal(800, 200, n_samples),
                'Temperature_C': np.random.normal(25, 5, n_samples),
                'Soil_Type': np.random.choice(['Sandy', 'Clay', 'Loamy'], n_samples),
                'Depth_m': np.random.uniform(10, 100, n_samples),
                'Water_Level_m': np.random.normal(15, 3, n_samples)
            }
            
        elif dataset_type == 'timeseries':
            dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
            trend = np.linspace(10, 15, n_samples)
            seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
            noise = np.random.normal(0, 0.5, n_samples)
            
            data = {
                'Date': dates,
                'Water_Level_m': trend + seasonal + noise,
                'Station_ID': ['ST001'] * n_samples
            }
            
        elif dataset_type == 'geospatial':
            # Generate coordinates around India
            base_lat, base_lon = 20.5937, 78.9629
            data = {
                'Station_ID': [f'ST{i:03d}' for i in range(n_samples)],
                'Latitude': np.random.normal(base_lat, 5, n_samples),
                'Longitude': np.random.normal(base_lon, 8, n_samples),
                'Water_Level_m': np.random.normal(15, 4, n_samples),
                'Depth_m': np.random.uniform(20, 80, n_samples)
            }
        
        return pd.DataFrame(data)
