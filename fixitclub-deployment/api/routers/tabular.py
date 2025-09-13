"""
Tabular analysis router for regression and classification tasks.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import os
import uuid

from ..schemas import TabularRequest, TabularResponse
from models.tabular import train_and_predict
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

@router.post("/tabular", response_model=TabularResponse)
async def analyze_tabular(
    dataset_id: Optional[str] = None,
    target_column: Optional[str] = None,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform tabular prediction analysis (regression or classification).
    
    Can use either a previously uploaded dataset (via dataset_id) or 
    upload a new file directly.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Run tabular analysis
        result = train_and_predict(
            df=df,
            target_column=target_column,
            task_type=task_type,
            test_size=test_size,
            random_state=random_state,
            save_model=True
        )
        
        # Check for errors in result
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create response
        response_data = {
            "task": result.get('task', 'tabular_prediction'),
            "target": result.get('target'),
            "metrics": result.get('metrics', {}),
            "feature_importance": result.get('feature_importance', {}),
            "sample_predictions": result.get('sample_predictions', []),
            "n_features": result.get('n_features'),
            "n_samples": result.get('n_samples'),
            "classes": result.get('classes'),  # For classification
            "model_path": result.get('model_path')
        }
        
        # Generate model ID from path if available
        model_id = None
        if result.get('model_path'):
            model_id = os.path.basename(result['model_path']).replace('.joblib', '')
        
        return TabularResponse(
            data=response_data,
            metrics=result.get('metrics'),
            model_id=model_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tabular analysis failed: {str(e)}")

@router.post("/tabular/predict")
async def predict_tabular(
    model_id: str,
    dataset_id: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Make predictions using a previously trained tabular model.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Load model and make predictions
        from models.tabular import load_and_predict
        
        model_path = f"saved_models/{model_id}.joblib"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Remove target column if it exists (for prediction)
        # This is a simple approach - in production you'd want better handling
        result = load_and_predict(model_path, df)
        
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "model_id": model_id,
            "predictions": result.get('predictions', []),
            "task_type": result.get('task_type'),
            "n_predictions": result.get('n_predictions'),
            "probabilities": result.get('probabilities'),
            "classes": result.get('classes')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")