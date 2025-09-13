"""
Anomaly detection router for outlier identification.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, List
import os

from ..schemas import AnomalyRequest, AnomalyResponse
from models.anomaly import detect, detect_temporal_anomalies
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

@router.post("/anomaly", response_model=AnomalyResponse)
async def analyze_anomaly(
    dataset_id: Optional[str] = None,
    contamination: float = 0.1,
    method: str = "isolation_forest",
    features: Optional[List[str]] = None,
    scale_features: bool = True,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform anomaly detection on dataset.
    
    Can use either a previously uploaded dataset (via dataset_id) or 
    upload a new file directly.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate features if provided
        if features:
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Features not found: {missing_features}"
                )
        
        # Run anomaly detection
        result = detect(
            df=df,
            contamination=contamination,
            method=method,
            features=features,
            scale_features=scale_features
        )
        
        # Check for errors in result
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create response
        response_data = {
            "task": result.get('task', 'anomaly_detection'),
            "method": result.get('method'),
            "contamination": result.get('contamination'),
            "features_used": result.get('features_used', []),
            "anomaly_flags": result.get('anomaly_flags', []),
            "anomaly_scores": result.get('anomaly_scores', []),
            "statistics": result.get('statistics', {}),
            "feature_analysis": result.get('feature_analysis', {}),
            "anomaly_indices": result.get('anomaly_indices', []),
            "top_anomalies": result.get('top_anomalies', [])
        }
        
        return AnomalyResponse(
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.post("/anomaly/temporal")
async def analyze_temporal_anomaly(
    date_column: str,
    value_column: str,
    dataset_id: Optional[str] = None,
    window_size: int = 30,
    threshold: float = 3.0,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform temporal anomaly detection using rolling statistics.
    
    Detects anomalies in time series data by identifying values that deviate
    significantly from the rolling mean.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{date_column}' not found")
        if value_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{value_column}' not found")
        
        # Run temporal anomaly detection
        result = detect_temporal_anomalies(
            df=df,
            date_column=date_column,
            value_column=value_column,
            window_size=window_size,
            threshold=threshold
        )
        
        # Check for errors in result
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal anomaly detection failed: {str(e)}")

@router.post("/anomaly/batch")
async def batch_anomaly_detection(
    datasets: List[str],
    contamination: float = 0.1,
    method: str = "isolation_forest"
):
    """
    Run anomaly detection on multiple datasets.
    
    Useful for batch processing of multiple groundwater station datasets.
    """
    try:
        results = {}
        
        for dataset_id in datasets:
            try:
                # Load dataset
                df = await load_dataset(dataset_id, None)
                
                # Run anomaly detection
                result = detect(
                    df=df,
                    contamination=contamination,
                    method=method
                )
                
                if result.get('success', True):
                    results[dataset_id] = {
                        "success": True,
                        "anomalies_detected": result.get('statistics', {}).get('anomalies_detected', 0),
                        "anomaly_rate": result.get('statistics', {}).get('anomaly_rate', 0),
                        "top_anomalies": result.get('top_anomalies', [])[:5]  # Top 5 only
                    }
                else:
                    results[dataset_id] = {
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    }
                    
            except Exception as e:
                results[dataset_id] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Summary statistics
        successful_analyses = sum(1 for r in results.values() if r.get('success'))
        total_anomalies = sum(r.get('anomalies_detected', 0) for r in results.values() if r.get('success'))
        
        return {
            "success": True,
            "summary": {
                "total_datasets": len(datasets),
                "successful_analyses": successful_analyses,
                "failed_analyses": len(datasets) - successful_analyses,
                "total_anomalies_detected": total_anomalies
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch anomaly detection failed: {str(e)}")