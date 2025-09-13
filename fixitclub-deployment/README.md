# Overview

This is a modular AI/ML platform specifically designed for groundwater analysis in India, supporting 5,260+ DWLR (Digital Water Level Recorder) stations. The platform automatically detects dataset types and performs various ML tasks including tabular prediction, time-series forecasting, anomaly detection, and geospatial mapping. Built with Streamlit for the web interface, it accepts CSV/Excel uploads and provides structured JSON outputs for easy integration with other systems.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based web interface** with a multi-page layout (Home, Dataset Upload & Analysis, Example Tests, Model Documentation)
- **Responsive design** with sidebar navigation and wide layout configuration
- **Interactive data visualization** using Plotly Express and Plotly Graph Objects for charts and maps
- **File upload handling** supporting CSV and Excel formats with automatic delimiter detection

## Backend Architecture
- **Modular design** with separate packages for models, utilities, and examples
- **DataProcessor class** handles file loading, preprocessing, and automatic dataset type detection (tabular, time-series, geospatial)
- **Four specialized ML modules**:
  - `tabular.py`: RandomForest for regression/classification with feature importance
  - `timeseries.py`: Holt-Winters and SARIMAX for seasonal forecasting with decomposition
  - `anomaly.py`: Isolation Forest and Local Outlier Factor for outlier detection
  - `geospatial.py`: KMeans clustering and coordinate-based analysis with GeoJSON output
- **Validation system** for data quality checks and input validation
- **Automatic preprocessing** including missing value handling, categorical encoding, and feature scaling

## Data Processing Pipeline
- **Multi-delimiter support** (`,` and `;`) with automatic detection
- **Missing value imputation** using SimpleImputer strategies
- **Categorical feature encoding** via OneHotEncoding and LabelEncoder
- **Date pattern recognition** supporting multiple formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Automatic target column detection** with manual override option
- **Coordinate detection** for enabling geospatial processing

## Model Architecture
- **Scikit-learn based models** with consistent interfaces across all modules
- **Standardized output format** returning JSON/dict objects for easy consumption
- **Feature importance extraction** for interpretability
- **Model persistence** with optional saving using joblib
- **Configurable hyperparameters** for all model types
- **Error handling and validation** at each processing step

# External Dependencies

## Core ML Libraries
- **scikit-learn**: RandomForest, clustering, anomaly detection, preprocessing, and metrics
- **statsmodels**: Time-series analysis (Holt-Winters, SARIMAX, seasonal decomposition)
- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and analysis

## Visualization and UI
- **Streamlit**: Web application framework for the user interface
- **Plotly Express & Graph Objects**: Interactive plotting and geospatial visualization
- **plotly.subplots**: Multi-panel chart creation

## Data Processing
- **joblib**: Model serialization and persistence
- **io.StringIO**: In-memory string operations for file processing

## Geospatial Processing
- **haversine_distances**: Distance calculations between geographic coordinates
- **GeoJSON format support**: For mapping and spatial data visualization

## Development and Testing
- **warnings**: Error suppression for cleaner output
- **traceback**: Error debugging and logging
- **typing**: Type hints for better code documentation