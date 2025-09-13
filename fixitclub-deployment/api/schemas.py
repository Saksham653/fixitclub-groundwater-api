"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Base response schemas
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]

class BaseResponse(BaseModel):
    success: bool = True
    task: str
    data: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    warnings: Optional[List[str]] = None
    model_id: Optional[str] = None
    trace_id: Optional[str] = None

# Dataset schemas
class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    shape: List[int]
    columns: List[str]
    detected_type: str
    features: Dict[str, str]
    upload_time: datetime

class DatasetUploadResponse(BaseModel):
    success: bool = True
    data: DatasetInfo

# Tabular analysis schemas
class TabularRequest(BaseModel):
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    task_type: str = Field(default="auto", description="auto, regression, or classification")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)

class TabularResponse(BaseResponse):
    task: str = "tabular_prediction"

# Time-series analysis schemas
class TimeSeriesRequest(BaseModel):
    dataset_id: Optional[str] = None
    date_column: str
    value_column: str
    seasonal_periods: int = Field(default=12, ge=1)
    horizon: int = Field(default=6, ge=1)
    method: str = Field(default="holt_winters", description="holt_winters or sarimax")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)

class TimeSeriesResponse(BaseResponse):
    task: str = "timeseries_forecast"

# Anomaly detection schemas
class AnomalyRequest(BaseModel):
    dataset_id: Optional[str] = None
    contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    method: str = Field(default="isolation_forest", description="isolation_forest or local_outlier_factor")
    features: Optional[List[str]] = None
    scale_features: bool = True

class AnomalyResponse(BaseResponse):
    task: str = "anomaly_detection"

# Geospatial analysis schemas
class GeospatialRequest(BaseModel):
    dataset_id: Optional[str] = None
    lat_column: str
    lon_column: str
    value_column: Optional[str] = None
    station_id_column: Optional[str] = None
    cluster_stations: bool = False
    n_clusters: int = Field(default=5, ge=2, le=20)

class GeospatialResponse(BaseResponse):
    task: str = "geospatial_mapping"

# Detection schemas
class DetectionRequest(BaseModel):
    dataset_id: str

class DetectionResponse(BaseModel):
    success: bool = True
    dataset_id: str
    detected_type: str
    features: Dict[str, str]
    recommendations: List[str]

# Model prediction schemas
class PredictionRequest(BaseModel):
    model_id: str
    data: Union[List[Dict[str, Any]], Dict[str, Any]]

class PredictionResponse(BaseModel):
    success: bool = True
    model_id: str
    predictions: List[Any]
    task_type: str

# Auto analysis schema
class AutoAnalysisRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None

class AutoAnalysisResponse(BaseModel):
    success: bool = True
    dataset_id: str
    analyses_performed: List[str]
    results: Dict[str, Any]