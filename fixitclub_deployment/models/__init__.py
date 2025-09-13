"""
AI/ML Models package for groundwater analysis.
Contains modular implementations for different analysis types.
"""

from .tabular import train_and_predict
from .timeseries import forecast
from .anomaly import detect
from .geospatial import map_data

__all__ = ['train_and_predict', 'forecast', 'detect', 'map_data']
