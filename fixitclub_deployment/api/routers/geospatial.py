"""
Geospatial analysis router for mapping and clustering tasks.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
import os

from ..schemas import GeospatialRequest, GeospatialResponse
from models.geospatial import map_data, find_similar_stations, generate_station_heatmap_data
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

@router.post("/geospatial", response_model=GeospatialResponse)
async def analyze_geospatial(
    lat_column: str,
    lon_column: str,
    dataset_id: Optional[str] = None,
    value_column: Optional[str] = None,
    station_id_column: Optional[str] = None,
    cluster_stations: bool = False,
    n_clusters: int = 5,
    file: Optional[UploadFile] = File(None)
):
    """
    Perform geospatial mapping and analysis.
    
    Can use either a previously uploaded dataset (via dataset_id) or 
    upload a new file directly.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        if lat_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Latitude column '{lat_column}' not found")
        if lon_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Longitude column '{lon_column}' not found")
        
        # Validate optional columns
        if value_column and value_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{value_column}' not found")
        if station_id_column and station_id_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Station ID column '{station_id_column}' not found")
        
        # Run geospatial analysis
        result = map_data(
            df=df,
            lat_column=lat_column,
            lon_column=lon_column,
            value_column=value_column,
            station_id_column=station_id_column,
            cluster_stations=cluster_stations,
            n_clusters=n_clusters
        )
        
        # Check for errors in result
        if not result.get('success', True) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create response
        response_data = {
            "task": result.get('task', 'geospatial_mapping'),
            "total_stations": result.get('total_stations'),
            "stations": result.get('stations', []),
            "geojson": result.get('geojson'),
            "bounds": result.get('bounds'),
            "center": result.get('center'),
            "heatmap_data": result.get('heatmap_data', []),
            "clustering": result.get('clustering')
        }
        
        return GeospatialResponse(
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geospatial analysis failed: {str(e)}")

@router.post("/geospatial/similar")
async def find_similar_stations_endpoint(
    lat_column: str,
    lon_column: str,
    value_column: str,
    target_station_id: str,
    dataset_id: Optional[str] = None,
    k: int = 5,
    file: Optional[UploadFile] = File(None)
):
    """
    Find stations with similar patterns to a target station.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        required_columns = [lat_column, lon_column, value_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Required columns not found: {missing_columns}"
            )
        
        # Run similarity analysis
        result = find_similar_stations(
            df=df,
            lat_column=lat_column,
            lon_column=lon_column,
            value_column=value_column,
            target_station_id=target_station_id,
            k=k
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
        raise HTTPException(status_code=500, detail=f"Similar stations analysis failed: {str(e)}")

@router.post("/geospatial/heatmap")
async def generate_heatmap(
    lat_column: str,
    lon_column: str,
    value_column: str,
    dataset_id: Optional[str] = None,
    grid_size: int = 20,
    file: Optional[UploadFile] = File(None)
):
    """
    Generate heatmap intensity data for station visualization.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate required columns
        required_columns = [lat_column, lon_column, value_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Required columns not found: {missing_columns}"
            )
        
        # Generate heatmap data
        result = generate_station_heatmap_data(
            df=df,
            lat_column=lat_column,
            lon_column=lon_column,
            value_column=value_column,
            grid_size=grid_size
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
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")

@router.post("/geospatial/validate-coordinates")
async def validate_coordinates(
    lat_column: str,
    lon_column: str,
    dataset_id: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Validate coordinate data quality and provide statistics.
    """
    try:
        # Load dataset
        df = await load_dataset(dataset_id, file)
        
        # Validate coordinates using validator
        from utils.validators import validate_coordinate_columns
        
        validation_result = validate_coordinate_columns(df, lat_column, lon_column)
        
        return {
            "success": validation_result['is_valid'],
            "validation_result": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coordinate validation failed: {str(e)}")