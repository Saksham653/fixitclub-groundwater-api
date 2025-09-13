"""
Time-series analysis router for forecasting tasks.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
import os

from ..schemas import TimeSeriesRequest, TimeSeriesResponse
from models.timeseries import forecast
from utils.data_processor import DataProcessor

router = APIRouter()
processor = DataProcessor()

async def load_dataset(dataset_id: Optional[str], file: Optional[UploadFile]):
    """Load dataset from ID or uploaded file"""
    if dataset_id:
        # Load from saved dataset
        file_path = None
        for ext in ['.csv', '.xlsx', '.xls']:
            potential_path = f"uploads/{dataset_id}{ext}"
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        class TempFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self._file = open(path, 'rb')
            
            def read(self, size=-1):
                return self._file.read(size)
            
            def seek(self, offset):
                return self._file.seek(offset)
            
            def close(self):
                return self._file.close()
        
        temp_file = TempFile(file_path)
        df = processor.load_data(temp_file)
        temp_file.close()
        return df
        
    elif file:
        # Load from uploaded file
        return processor.load_data(file)
    
    else:
        raise HTTPException(status_code=400, detail="Either dataset_id or file must be provided")

@router.post("/timeseries", response_model=TimeSeriesResponse)
async def analyze_timeseries(
    date_column: str,
    value_column: str,
    dataset_id: Optional[str] = None,
    seasonal_periods: int = 12,
    horizon: int = 6,
    method: str = "holt_winters",
    confidence_level: float = 0.95,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform time-series forecasting analysis.
    
    Can use either a previously uploaded dataset (via dataset_id) or 
    upload a new file directly.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{date_column}' not found")
        if value_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{value_column}' not found")
        
        # Run time-series analysis
        result = forecast(
            df=df,
            date_column=date_column,
            value_column=value_column,
            seasonal_periods=seasonal_periods,
            horizon=horizon,
            method=method,
            confidence_level=confidence_level
        )
        
        # Check for errors in result
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create response
        response_data = {
            "task": result.get('task', 'timeseries_forecast'),
            "method": result.get('method'),
            "seasonal_periods": result.get('seasonal_periods'),
            "horizon": result.get('horizon'),
            "forecast_values": result.get('forecast_values', []),
            "confidence_intervals": result.get('confidence_intervals', []),
            "components": result.get('components', {}),
            "last_observed_value": result.get('last_observed_value'),
            "forecast_dates": result.get('forecast_dates', [])
        }
        
        return TimeSeriesResponse(
            data=response_data,
            metrics=result.get('metrics')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time-series analysis failed: {str(e)}")

@router.post("/timeseries/decompose")
async def decompose_timeseries(
    date_column: str,
    value_column: str,
    dataset_id: Optional[str] = None,
    seasonal_periods: int = 12,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform seasonal decomposition of time series data.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{date_column}' not found")
        if value_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{value_column}' not found")
        
        # Prepare time series data
        import pandas as pd
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        ts_df = df[[date_column, value_column]].copy()
        ts_df = ts_df.dropna()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        ts_df = ts_df.sort_values(date_column)
        ts_df.set_index(date_column, inplace=True)
        
        # Handle duplicates by taking mean
        ts_df = ts_df.groupby(ts_df.index).mean()
        ts = ts_df[value_column]
        
        if len(ts) < seasonal_periods * 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data for seasonal decomposition. Need at least {seasonal_periods * 2} points."
            )
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts, model='additive', period=seasonal_periods)
        
        # Prepare results
        result = {
            "task": "timeseries_decomposition",
            "seasonal_periods": seasonal_periods,
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "residual": decomposition.resid.dropna().tolist(),
            "observed": decomposition.observed.tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in decomposition.trend.dropna().index]
        }
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time-series decomposition failed: {str(e)}")