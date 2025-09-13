"""
Anomaly detection models for identifying unusual groundwater patterns.
Supports Isolation Forest and Local Outlier Factor methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def detect(df, contamination=0.1, method='isolation_forest', features=None, scale_features=True):
    """
    Detect anomalies in groundwater data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    contamination : float
        Expected proportion of outliers (0.01 to 0.5)
    method : str
        'isolation_forest' or 'local_outlier_factor'
    features : list, optional
        Specific features to use for detection. If None, uses all numeric features
    scale_features : bool
        Whether to standardize features before detection
    
    Returns:
    --------
    dict : Anomaly detection results with scores and flags
    """
    
    try:
        # Input validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if not 0.01 <= contamination <= 0.5:
            contamination = 0.1  # Default fallback
        
        # Select features for anomaly detection
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError("No numeric features found for anomaly detection")
            features = numeric_cols
        else:
            # Validate provided features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Features not found: {missing_features}")
        
        # Prepare feature matrix
        X = df[features].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )
        else:
            X_scaled = X_imputed
        
        # Perform anomaly detection
        if method == 'isolation_forest':
            anomaly_result = _isolation_forest_detection(X_scaled, contamination)
        elif method == 'local_outlier_factor':
            anomaly_result = _lof_detection(X_scaled, contamination)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate additional statistics
        anomaly_indices = [i for i, flag in enumerate(anomaly_result['anomaly_flags']) if flag]
        normal_indices = [i for i, flag in enumerate(anomaly_result['anomaly_flags']) if not flag]
        
        # Statistical summary
        stats = {
            'total_samples': len(df),
            'anomalies_detected': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(df),
            'mean_anomaly_score': np.mean([anomaly_result['anomaly_scores'][i] for i in anomaly_indices]) if anomaly_indices else 0,
            'mean_normal_score': np.mean([anomaly_result['anomaly_scores'][i] for i in normal_indices]) if normal_indices else 0
        }
        
        # Feature-wise anomaly analysis
        feature_anomaly_stats = {}
        if anomaly_indices:
            for feature in features:
                anomaly_values = df.iloc[anomaly_indices][feature].dropna()
                normal_values = df.iloc[normal_indices][feature].dropna()
                
                if len(anomaly_values) > 0 and len(normal_values) > 0:
                    feature_anomaly_stats[feature] = {
                        'anomaly_mean': anomaly_values.mean(),
                        'normal_mean': normal_values.mean(),
                        'difference_ratio': abs(anomaly_values.mean() - normal_values.mean()) / normal_values.std() if normal_values.std() > 0 else 0
                    }
        
        # Prepare result
        result = {
            'task': 'anomaly_detection',
            'method': method,
            'contamination': contamination,
            'features_used': features,
            'anomaly_flags': anomaly_result['anomaly_flags'],
            'anomaly_scores': anomaly_result['anomaly_scores'],
            'statistics': stats,
            'feature_analysis': feature_anomaly_stats,
            'anomaly_indices': anomaly_indices
        }
        
        # Add top anomalies
        if anomaly_indices:
            # Sort by anomaly score
            sorted_anomalies = sorted(
                [(i, anomaly_result['anomaly_scores'][i]) for i in anomaly_indices],
                key=lambda x: x[1],
                reverse=True
            )
            
            result['top_anomalies'] = [
                {
                    'index': idx,
                    'anomaly_score': score,
                    'values': df.iloc[idx][features].to_dict()
                }
                for idx, score in sorted_anomalies[:10]  # Top 10
            ]
        
        return result
        
    except Exception as e:
        return {
            'task': 'anomaly_detection',
            'error': str(e),
            'success': False
        }

def _isolation_forest_detection(X, contamination):
    """Isolation Forest anomaly detection"""
    
    try:
        # Initialize Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit and predict
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Convert labels (-1 for anomaly, 1 for normal) to boolean flags
        anomaly_flags = (anomaly_labels == -1).tolist()
        
        # Normalize scores to [0, 1] range for consistency
        # Higher scores indicate more anomalous
        normalized_scores = []
        score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
        
        for score in anomaly_scores:
            if score_max != score_min:
                # Invert and normalize: lower isolation scores = more anomalous
                normalized_score = (score_max - score) / (score_max - score_min)
            else:
                normalized_score = 0.5
            normalized_scores.append(normalized_score)
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': normalized_scores
        }
        
    except Exception as e:
        raise Exception(f"Isolation Forest failed: {str(e)}")

def _lof_detection(X, contamination):
    """Local Outlier Factor anomaly detection"""
    
    try:
        # Calculate number of neighbors (min 5, max 50)
        n_neighbors = min(max(5, int(len(X) * 0.05)), 50)
        n_neighbors = min(n_neighbors, len(X) - 1)
        
        if n_neighbors < 2:
            raise ValueError("Not enough samples for LOF")
        
        # Initialize LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        # Fit and predict
        anomaly_labels = lof.fit_predict(X)
        anomaly_scores = lof.negative_outlier_factor_
        
        # Convert labels (-1 for anomaly, 1 for normal) to boolean flags
        anomaly_flags = (anomaly_labels == -1).tolist()
        
        # Normalize scores to [0, 1] range
        # LOF scores are negative, more negative = more anomalous
        normalized_scores = []
        score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
        
        for score in anomaly_scores:
            if score_max != score_min:
                # Invert and normalize: more negative LOF scores = more anomalous
                normalized_score = (score_max - score) / (score_max - score_min)
            else:
                normalized_score = 0.5
            normalized_scores.append(normalized_score)
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': normalized_scores
        }
        
    except Exception as e:
        raise Exception(f"LOF failed: {str(e)}")

def detect_temporal_anomalies(df, date_column, value_column, window_size=30, threshold=3):
    """
    Detect temporal anomalies using rolling statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset with time series data
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column
    window_size : int
        Rolling window size for statistics
    threshold : float
        Number of standard deviations for anomaly threshold
    
    Returns:
    --------
    dict : Temporal anomaly detection results
    """
    
    try:
        # Prepare time series
        ts_df = df[[date_column, value_column]].copy()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        ts_df = ts_df.sort_values(date_column).dropna()
        
        if len(ts_df) < window_size:
            raise ValueError(f"Not enough data points for window size {window_size}")
        
        # Calculate rolling statistics
        ts_df['rolling_mean'] = ts_df[value_column].rolling(window=window_size, center=True).mean()
        ts_df['rolling_std'] = ts_df[value_column].rolling(window=window_size, center=True).std()
        
        # Calculate z-scores
        ts_df['z_score'] = np.abs(
            (ts_df[value_column] - ts_df['rolling_mean']) / ts_df['rolling_std']
        )
        
        # Identify anomalies
        ts_df['is_anomaly'] = ts_df['z_score'] > threshold
        
        # Prepare results
        anomalies = ts_df[ts_df['is_anomaly']].copy()
        
        result = {
            'task': 'temporal_anomaly_detection',
            'window_size': window_size,
            'threshold': threshold,
            'total_points': len(ts_df),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(ts_df),
            'anomalies': [
                {
                    'date': row[date_column].strftime('%Y-%m-%d'),
                    'value': row[value_column],
                    'z_score': row['z_score'],
                    'expected_range': [
                        row['rolling_mean'] - threshold * row['rolling_std'],
                        row['rolling_mean'] + threshold * row['rolling_std']
                    ]
                }
                for _, row in anomalies.iterrows()
            ]
        }
        
        return result
        
    except Exception as e:
        return {
            'task': 'temporal_anomaly_detection',
            'error': str(e),
            'success': False
        }
