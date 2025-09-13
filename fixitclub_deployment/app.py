import streamlit as st
import pandas as pd
import numpy as np
import json
import traceback
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from models.tabular import train_and_predict
from models.timeseries import forecast
from models.anomaly import detect
from models.geospatial import map_data
from utils.data_processor import DataProcessor
from utils.validators import validate_dataset
from examples.test_functions import run_example_tests

st.set_page_config(
    page_title="Groundwater AI/ML Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("💧 Groundwater AI/ML Analysis Platform")
    st.markdown("Upload your groundwater data and get AI-powered insights with automatic dataset detection")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Home", "Dataset Upload & Analysis", "Example Tests", "Model Documentation"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Dataset Upload & Analysis":
        show_analysis_page()
    elif page == "Example Tests":
        show_example_tests()
    elif page == "Model Documentation":
        show_documentation()

def show_home_page():
    st.header("Welcome to the Groundwater Analysis Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Supported Analysis Types")
        st.markdown("""
        - **Tabular Prediction**: Regression and classification with feature importance
        - **Time-series Forecasting**: Seasonal groundwater level prediction
        - **Anomaly Detection**: Identify unusual water level patterns
        - **Geospatial Mapping**: Station-based mapping with GeoJSON output
        - **Alert Generation**: Threshold-based risk assessment
        """)
    
    with col2:
        st.subheader("📊 Key Features")
        st.markdown("""
        - **Automatic Dataset Detection**: Tabular, time-series, or geospatial
        - **Smart Preprocessing**: Handle missing values, mixed delimiters
        - **Model Persistence**: Save and reuse trained models
        - **JSON/GeoJSON Output**: Ready for UI integration
        - **Interactive Visualizations**: Powered by Plotly
        """)
    
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. Navigate to **Dataset Upload & Analysis** in the sidebar
    2. Upload your CSV or Excel file
    3. The system will automatically detect the dataset type
    4. Choose your analysis type and configure parameters
    5. View results and download JSON outputs
    """)

def show_analysis_page():
    st.header("📁 Dataset Upload & Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your groundwater dataset",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            processor = DataProcessor()
            df = processor.load_data(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Dataset type detection
            dataset_info = processor.detect_dataset_type(df)
            
            st.subheader("🔍 Dataset Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Detected Type**: {dataset_info['type'].title()}")
                if dataset_info['features']:
                    st.write("**Key Features Detected:**")
                    for feature, details in dataset_info['features'].items():
                        st.write(f"- {feature}: {details}")
            
            with col2:
                st.write("**Column Info:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
            
            # Analysis selection based on detected type
            st.subheader("🎯 Select Analysis Type")
            
            analysis_options = ["Tabular Prediction"]
            if dataset_info['features'].get('temporal'):
                analysis_options.append("Time-series Forecasting")
            if dataset_info['features'].get('geospatial'):
                analysis_options.append("Geospatial Mapping")
            analysis_options.extend(["Anomaly Detection", "Alert Generation"])
            
            analysis_type = st.selectbox("Choose analysis:", analysis_options)
            
            # Run analysis based on selection
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner(f"Running {analysis_type}..."):
                    run_analysis(df, analysis_type, dataset_info)
                    
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.code(traceback.format_exc())

def run_analysis(df, analysis_type, dataset_info):
    """Run the selected analysis and display results"""
    
    try:
        if analysis_type == "Tabular Prediction":
            run_tabular_analysis(df, dataset_info)
        elif analysis_type == "Time-series Forecasting":
            run_timeseries_analysis(df, dataset_info)
        elif analysis_type == "Anomaly Detection":
            run_anomaly_analysis(df, dataset_info)
        elif analysis_type == "Geospatial Mapping":
            run_geospatial_analysis(df, dataset_info)
        elif analysis_type == "Alert Generation":
            run_alert_analysis(df, dataset_info)
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.code(traceback.format_exc())

def run_tabular_analysis(df, dataset_info):
    """Run tabular prediction analysis"""
    st.subheader("📊 Tabular Prediction Results")
    
    # Target selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = st.selectbox("Select target column:", numeric_cols)
    
    if target_col:
        # Task type selection
        task_type = st.radio("Task type:", ["regression", "classification"])
        
        # Run analysis
        result = train_and_predict(df, target_column=target_col, task_type=task_type)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Performance:**")
            for metric, value in result['metrics'].items():
                st.metric(metric.upper(), f"{value:.4f}")
        
        with col2:
            st.write("**Feature Importance:**")
            importance_df = pd.DataFrame(
                list(result['feature_importance'].items())
            )
            importance_df.columns = ['Feature', 'Importance']
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        if 'sample_predictions' in result:
            st.write("**Sample Predictions:**")
            pred_df = pd.DataFrame({
                'Actual': df[target_col].head(len(result['sample_predictions'])),
                'Predicted': result['sample_predictions']
            })
            fig = px.scatter(pred_df, x='Actual', y='Predicted')
            fig.add_shape(type="line", x0=pred_df['Actual'].min(), y0=pred_df['Actual'].min(),
                         x1=pred_df['Actual'].max(), y1=pred_df['Actual'].max(),
                         line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)
        
        # JSON output
        with st.expander("📄 JSON Output"):
            st.json(result)

def run_timeseries_analysis(df, dataset_info):
    """Run time-series forecasting analysis"""
    st.subheader("📈 Time-series Forecasting Results")
    
    # Column selection
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select date column:", date_cols + df.columns.tolist())
    with col2:
        value_col = st.selectbox("Select value column:", value_cols)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        seasonal_periods = st.number_input("Seasonal periods:", min_value=1, value=12)
    with col2:
        forecast_horizon = st.number_input("Forecast horizon:", min_value=1, value=6)
    with col3:
        method = st.selectbox("Method:", ["holt_winters", "sarimax"])
    
    if date_col and value_col:
        # Run forecast
        result = forecast(
            df, 
            date_column=date_col, 
            value_column=value_col,
            seasonal_periods=seasonal_periods,
            horizon=forecast_horizon,
            method=method
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Forecast Metrics:**")
            for metric, value in result['metrics'].items():
                st.metric(metric.upper(), f"{value:.4f}")
        
        with col2:
            st.write("**Model Components:**")
            if 'components' in result:
                for comp, desc in result['components'].items():
                    st.write(f"- {comp}: {desc}")
        
        # Forecast plot
        if 'forecast_values' in result:
            st.write("**Forecast Visualization:**")
            
            # Create time series plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df[date_col].tail(50),
                y=df[value_col].tail(50),
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            forecast_dates = pd.date_range(
                start=df[date_col].max(), 
                periods=forecast_horizon + 1,
                freq='D'
            )[1:]
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=result['forecast_values'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title="Time Series Forecast", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
        
        # JSON output
        with st.expander("📄 JSON Output"):
            st.json(result)

def run_anomaly_analysis(df, dataset_info):
    """Run anomaly detection analysis"""
    st.subheader("🚨 Anomaly Detection Results")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider("Contamination rate:", 0.01, 0.3, 0.1)
    with col2:
        method = st.selectbox("Method:", ["isolation_forest", "local_outlier_factor"])
    
    # Run detection
    result = detect(df, contamination=contamination, method=method)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        total_anomalies = sum(result['anomaly_flags'])
        st.metric("Total Anomalies", total_anomalies)
    with col2:
        anomaly_rate = total_anomalies / len(result['anomaly_flags']) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    with col3:
        avg_score = np.mean(result['anomaly_scores'])
        st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    
    # Anomaly visualization
    if len(df.select_dtypes(include=[np.number]).columns) >= 2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
        
        fig = px.scatter(
            x=df[numeric_cols[0]],
            y=df[numeric_cols[1]],
            color=result['anomaly_flags'],
            color_discrete_map={True: 'red', False: 'blue'},
            labels={'color': 'Anomaly'},
            title="Anomaly Detection Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show anomalous records
    if total_anomalies > 0:
        st.write("**Anomalous Records:**")
        anomaly_indices = [i for i, flag in enumerate(result['anomaly_flags']) if flag]
        anomaly_df = df.iloc[anomaly_indices].copy()
        anomaly_df['anomaly_score'] = [result['anomaly_scores'][i] for i in anomaly_indices]
        st.dataframe(anomaly_df)
    
    # JSON output
    with st.expander("📄 JSON Output"):
        st.json(result)

def run_geospatial_analysis(df, dataset_info):
    """Run geospatial mapping analysis"""
    st.subheader("🗺️ Geospatial Mapping Results")
    
    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        lat_col = st.selectbox("Latitude column:", df.columns)
    with col2:
        lon_col = st.selectbox("Longitude column:", df.columns)
    
    value_col = st.selectbox("Value column (optional):", ['None'] + df.select_dtypes(include=[np.number]).columns.tolist())
    
    if lat_col and lon_col:
        value_column = None if value_col == 'None' else value_col
        
        # Run mapping
        result = map_data(df, lat_column=lat_col, lon_column=lon_col, value_column=value_column)
        
        # Display results
        st.write(f"**Total Stations:** {len(result['stations'])}")
        
        # Create map visualization
        if result['stations']:
            stations_df = pd.DataFrame(result['stations'])
            
            if value_column:
                fig = px.scatter_mapbox(
                    stations_df,
                    lat='lat',
                    lon='lon',
                    color=value_column,
                    size=value_column,
                    hover_data=['id'] if 'id' in stations_df.columns else None,
                    mapbox_style="open-street-map",
                    title="Station Locations"
                )
            else:
                fig = px.scatter_mapbox(
                    stations_df,
                    lat='lat',
                    lon='lon',
                    hover_data=['id'] if 'id' in stations_df.columns else None,
                    mapbox_style="open-street-map",
                    title="Station Locations"
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # GeoJSON output
        with st.expander("🌐 GeoJSON Output"):
            if 'geojson' in result:
                st.json(result['geojson'])
        
        # JSON output
        with st.expander("📄 JSON Output"):
            st.json(result)

def run_alert_analysis(df, dataset_info):
    """Run alert generation analysis"""
    st.subheader("⚠️ Alert Generation Results")
    
    # Select value column
    value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = st.selectbox("Select value column for alerts:", value_cols)
    
    if value_col:
        # Threshold configuration
        col1, col2 = st.columns(2)
        with col1:
            upper_threshold = st.number_input(
                "Upper threshold:", 
                value=float(df[value_col].quantile(0.9))
            )
        with col2:
            lower_threshold = st.number_input(
                "Lower threshold:", 
                value=float(df[value_col].quantile(0.1))
            )
        
        # Generate alerts
        alerts = []
        risk_scores = []
        
        for idx, value in df[value_col].items():
            alert = None
            risk_score = 0
            
            if value > upper_threshold:
                alert = "HIGH"
                risk_score = min((value - upper_threshold) / upper_threshold, 1.0)
            elif value < lower_threshold:
                alert = "LOW" 
                risk_score = min((lower_threshold - value) / lower_threshold, 1.0)
            else:
                alert = "NORMAL"
                risk_score = 0
            
            alerts.append(alert)
            risk_scores.append(risk_score)
        
        # Results
        alert_counts = pd.Series(alerts).value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Alerts", alert_counts.get('HIGH', 0))
        with col2:
            st.metric("Low Alerts", alert_counts.get('LOW', 0))
        with col3:
            avg_risk = np.mean(risk_scores)
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        
        # Alert visualization
        fig = px.histogram(x=alerts, title="Alert Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert details
        alert_df = df.copy()
        alert_df['alert'] = alerts
        alert_df['risk_score'] = risk_scores
        
        high_risk = alert_df[alert_df['alert'].isin(['HIGH', 'LOW'])].sort_values('risk_score', ascending=False)
        if len(high_risk) > 0:
            st.write("**High Risk Records:**")
            st.dataframe(high_risk.head(10))
        
        # JSON output
        result = {
            "task": "alert_generation",
            "thresholds": {"upper": upper_threshold, "lower": lower_threshold},
            "summary": {
                "total_records": len(df),
                "high_alerts": alert_counts.get('HIGH', 0),
                "low_alerts": alert_counts.get('LOW', 0),
                "normal": alert_counts.get('NORMAL', 0),
                "avg_risk_score": avg_risk
            },
            "alerts": [{"index": i, "alert": alert, "risk_score": score} 
                      for i, (alert, score) in enumerate(zip(alerts, risk_scores))]
        }
        
        with st.expander("📄 JSON Output"):
            st.json(result)

def show_example_tests():
    """Show example tests page"""
    st.header("🧪 Example Tests")
    st.markdown("Test all ML modules with synthetic data to verify functionality")
    
    if st.button("🚀 Run All Example Tests", type="primary"):
        with st.spinner("Running example tests..."):
            try:
                results = run_example_tests()
                
                for test_name, result in results.items():
                    with st.expander(f"✅ {test_name}", expanded=True):
                        if result['success']:
                            st.success("Test passed!")
                            st.json(result['output'])
                        else:
                            st.error(f"Test failed: {result['error']}")
                            
            except Exception as e:
                st.error(f"Tests failed: {str(e)}")

def show_documentation():
    """Show model documentation"""
    st.header("📚 Model Documentation")
    
    with st.expander("🎯 Tabular Models", expanded=True):
        st.markdown("""
        **RandomForest Implementation**
        - Supports both regression and classification
        - Automatic feature importance calculation
        - Handles categorical variables with OneHot encoding
        - Returns structured metrics (RMSE, R², MAE for regression; Accuracy, F1 for classification)
        """)
    
    with st.expander("📈 Time-series Models", expanded=True):
        st.markdown("""
        **Holt-Winters Exponential Smoothing**
        - Triple exponential smoothing for trend and seasonality
        - Configurable seasonal periods
        - Automatic parameter optimization
        
        **SARIMAX**
        - Seasonal ARIMA with exogenous variables
        - Automatic order selection
        - Handles missing values and irregular frequencies
        """)
    
    with st.expander("🚨 Anomaly Detection Models", expanded=True):
        st.markdown("""
        **Isolation Forest**
        - Unsupervised outlier detection
        - Configurable contamination rate
        - Returns anomaly scores and binary flags
        
        **Local Outlier Factor**
        - Density-based anomaly detection
        - Considers local neighborhood density
        - Effective for datasets with varying densities
        """)
    
    with st.expander("🗺️ Geospatial Processing", expanded=True):
        st.markdown("""
        **Station Mapping**
        - Automatic lat/lon detection
        - GeoJSON output for web mapping
        - Station clustering and similarity analysis
        - Heatmap data generation
        
        **Supported Output Formats**
        - GeoJSON FeatureCollection
        - Structured JSON with station arrays
        - Lightweight format optimized for web rendering
        """)

if __name__ == "__main__":
    main()
