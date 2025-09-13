"""
Dataset management router for file uploads and processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import uuid
import os
import shutil
from datetime import datetime

from ..schemas import DatasetUploadResponse, DatasetInfo, DetectionRequest, DetectionResponse
from utils.data_processor import DataProcessor
from utils.validators import validate_dataset

router = APIRouter()

# Initialize data processor
processor = DataProcessor()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    dataset_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = f"uploads/{dataset_id}{file_ext}"
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return dataset_id, file_path

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file (CSV or Excel) for analysis.
    
    Returns dataset information including detected type and features.
    """
    try:
        # Validate file
        validate_file(file)
        
        # Save file
        dataset_id, file_path = await save_uploaded_file(file)
        
        # Load and process data
        with open(file_path, 'rb') as f:
            # Create a temporary file object for processing
            class TempFile:
                def __init__(self, path, filename):
                    self.name = filename
                    self._file = open(path, 'rb')
                
                def read(self, size=-1):
                    return self._file.read(size)
                
                def seek(self, offset):
                    return self._file.seek(offset)
                
                def close(self):
                    return self._file.close()
            
            temp_file = TempFile(file_path, file.filename)
            df = processor.load_data(temp_file)
            temp_file.close()
        
        # Validate dataset
        validation_result = validate_dataset(df)
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset: {'; '.join(validation_result['errors'])}"
            )
        
        # Detect dataset type and features
        dataset_info = processor.detect_dataset_type(df)
        
        # Create response
        dataset_data = DatasetInfo(
            dataset_id=dataset_id,
            filename=file.filename,
            shape=list(df.shape),
            columns=df.columns.tolist(),
            detected_type=dataset_info['type'],
            features=dataset_info['features'],
            upload_time=datetime.now()
        )
        
        return DatasetUploadResponse(data=dataset_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

@router.post("/detect", response_model=DetectionResponse)
async def detect_dataset_type(request: DetectionRequest):
    """
    Detect dataset type and provide analysis recommendations.
    """
    try:
        dataset_id = request.dataset_id
        
        # Find dataset file
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
        
        # Detect type and features
        dataset_info = processor.detect_dataset_type(df)
        
        # Generate recommendations
        recommendations = []
        if dataset_info['type'] == 'tabular':
            recommendations.append("Suitable for tabular prediction (regression/classification)")
            recommendations.append("Use /api/v1/analyze/tabular endpoint")
        elif dataset_info['type'] == 'timeseries':
            recommendations.append("Suitable for time-series forecasting")
            recommendations.append("Use /api/v1/analyze/timeseries endpoint")
        elif dataset_info['type'] == 'geospatial':
            recommendations.append("Suitable for geospatial mapping and clustering")
            recommendations.append("Use /api/v1/analyze/geospatial endpoint")
        
        recommendations.append("Can always run anomaly detection on any dataset")
        
        return DetectionResponse(
            dataset_id=dataset_id,
            detected_type=dataset_info['type'],
            features=dataset_info['features'],
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect dataset type: {str(e)}")

@router.get("/list")
async def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = []
        uploads_dir = "uploads"
        
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    dataset_id = os.path.splitext(filename)[0]
                    file_path = os.path.join(uploads_dir, filename)
                    file_stats = os.stat(file_path)
                    
                    datasets.append({
                        "dataset_id": dataset_id,
                        "filename": filename,
                        "size_bytes": file_stats.st_size,
                        "upload_time": datetime.fromtimestamp(file_stats.st_mtime)
                    })
        
        return {"success": True, "datasets": datasets}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    try:
        deleted = False
        for ext in ALLOWED_EXTENSIONS:
            file_path = f"uploads/{dataset_id}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True
                break
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {"success": True, "message": f"Dataset {dataset_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")