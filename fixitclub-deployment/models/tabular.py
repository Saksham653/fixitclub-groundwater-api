"""
Tabular prediction models for groundwater data analysis.
Supports both regression and classification tasks with feature importance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_and_predict(df, target_column=None, task_type='auto', test_size=0.2, random_state=42, save_model=True):
    """
    Train a RandomForest model for tabular prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_column : str, optional
        Target column name. If None, automatically detects the last numeric column
    task_type : str
        'regression', 'classification', or 'auto' for automatic detection
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    save_model : bool
        Whether to save the trained model
    
    Returns:
    --------
    dict : Results containing metrics, predictions, and feature importance
    """
    
    try:
        # Input validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Auto-detect target column if not provided
        if target_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for target")
            target_column = numeric_cols[-1]  # Use last numeric column
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values in target
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) == 0:
            raise ValueError("No valid target values found")
        
        # Auto-detect task type
        if task_type == 'auto':
            if y.dtype == 'object' or len(y.unique()) < 10:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Encode categorical target for classification
        label_encoder = None
        if task_type == 'classification' and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Fit preprocessor and transform data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = numeric_features.copy()
        if categorical_features:
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
        )
        
        # Initialize model
        if task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if task_type == 'regression':
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        
        # Feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Sample predictions (first 10)
        sample_predictions = y_pred[:10].tolist()
        
        # Save model if requested
        model_path = None
        if save_model:
            os.makedirs('saved_models', exist_ok=True)
            model_path = f'saved_models/tabular_{task_type}_model.joblib'
            joblib.dump({
                'model': model,
                'preprocessor': preprocessor,
                'label_encoder': label_encoder,
                'feature_names': feature_names,
                'task_type': task_type
            }, model_path)
        
        # Prepare result
        result = {
            'task': f'tabular_{task_type}',
            'target': target_column,
            'metrics': metrics,
            'sample_predictions': sample_predictions,
            'feature_importance': feature_importance,
            'model_path': model_path,
            'n_features': len(feature_names),
            'n_samples': len(y)
        }
        
        # Add class information for classification
        if task_type == 'classification':
            if label_encoder:
                result['classes'] = label_encoder.classes_.tolist()
            else:
                result['classes'] = sorted(y.unique().tolist())
        
        return result
        
    except Exception as e:
        return {
            'task': 'tabular_prediction',
            'error': str(e),
            'success': False
        }

def load_and_predict(model_path, new_data):
    """
    Load a saved model and make predictions on new data.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    new_data : pandas.DataFrame
        New data for prediction
    
    Returns:
    --------
    dict : Predictions and metadata
    """
    
    try:
        # Load saved model
        saved_model = joblib.load(model_path)
        model = saved_model['model']
        preprocessor = saved_model['preprocessor']
        label_encoder = saved_model.get('label_encoder')
        
        # Preprocess new data
        X_processed = preprocessor.transform(new_data)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        # Decode predictions if classification with label encoding
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)
        
        # Get prediction probabilities for classification
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_processed)
        
        result = {
            'predictions': predictions.tolist(),
            'task_type': saved_model['task_type'],
            'n_predictions': len(predictions)
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities.tolist()
            result['classes'] = saved_model.get('classes', [])
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }
