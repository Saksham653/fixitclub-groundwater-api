"""
FastAPI application for Groundwater ML Analysis Platform.
Provides REST API endpoints for all ML analysis tasks.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import os
import uuid
import traceback
from typing import Optional

from .routers import tabular, timeseries, anomaly, geospatial, datasets
from .schemas import ErrorResponse, HealthResponse
from utils.data_processor import DataProcessor

# Initialize FastAPI app
app = FastAPI(
    title="FixItClub Groundwater ML Analysis API",
    description="FixItClub's AI/ML platform for analyzing groundwater data from 5,260+ DWLR stations",
    version="1.0.0",
    docs_url="/fixitclub/docs",
    redoc_url="/fixitclub/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple API key validation (optional)"""
    if credentials is None:
        return None
    # Add your API key validation logic here
    return credentials.credentials

# Include routers with fixitclub branding
app.include_router(datasets.router, prefix="/fixitclub/api/v1/datasets", tags=["FixItClub-datasets"])
app.include_router(tabular.router, prefix="/fixitclub/api/v1/analyze", tags=["FixItClub-tabular"])
app.include_router(timeseries.router, prefix="/fixitclub/api/v1/analyze", tags=["FixItClub-timeseries"])
app.include_router(anomaly.router, prefix="/fixitclub/api/v1/analyze", tags=["FixItClub-anomaly"])
app.include_router(geospatial.router, prefix="/fixitclub/api/v1/analyze", tags=["FixItClub-geospatial"])

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="ok",
        message="Welcome to FixItClub Groundwater ML Analysis API",
        version="1.0.0"
    )

@app.get("/fixitclub", response_model=HealthResponse)
async def fixitclub_root():
    """FixItClub branded root endpoint"""
    return HealthResponse(
        status="ok",
        message="FixItClub Groundwater ML Analysis API - Ready to serve!",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        message="FixItClub API is running perfectly!",
        version="1.0.0"
    )

@app.get("/fixitclub/health", response_model=HealthResponse)
async def fixitclub_health_check():
    """FixItClub branded health check endpoint"""
    return HealthResponse(
        status="ok",
        message="FixItClub Groundwater ML API - All systems operational!",
        version="1.0.0"
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    trace_id = str(uuid.uuid4())
    
    # Log the error (you can add proper logging here)
    print(f"Error {trace_id}: {str(exc)}")
    print(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "trace_id": trace_id
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )