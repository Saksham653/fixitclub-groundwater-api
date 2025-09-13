"""
Geospatial mapping and analysis for groundwater station data.
Handles coordinate-based data with clustering and similarity analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

def map_data(df, lat_column, lon_column, value_column=None, station_id_column=None, cluster_stations=False, n_clusters=5):
    """
    Process geospatial groundwater data for mapping.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset with coordinates
    lat_column : str
        Name of latitude column
    lon_column : str
        Name of longitude column  
    value_column : str, optional
        Name of value column for color coding
    station_id_column : str, optional
        Name of station ID column
    cluster_stations : bool
        Whether to perform station clustering
    n_clusters : int
        Number of clusters for station grouping
    
    Returns:
    --------
    dict : Geospatial data with stations and optional GeoJSON
    """
    
    try:
        # Input validation
        if lat_column not in df.columns:
            raise ValueError(f"Latitude column '{lat_column}' not found")
        if lon_column not in df.columns:
            raise ValueError(f"Longitude column '{lon_column}' not found")
        
        # Prepare geospatial data
        geo_df = df[[lat_column, lon_column]].copy()
        
        # Add value column if specified
        if value_column and value_column in df.columns:
            geo_df[value_column] = df[value_column]
        
        # Add station ID if specified
        if station_id_column and station_id_column in df.columns:
            geo_df['station_id'] = df[station_id_column]
        else:
            geo_df['station_id'] = [f'ST{i:03d}' for i in range(len(geo_df))]
        
        # Remove rows with invalid coordinates
        geo_df = geo_df.dropna(subset=[lat_column, lon_column])
        
        if len(geo_df) == 0:
            raise ValueError("No valid coordinate data found")
        
        # Validate coordinate ranges
        lat_valid = (-90 <= geo_df[lat_column]) & (geo_df[lat_column] <= 90)
        lon_valid = (-180 <= geo_df[lon_column]) & (geo_df[lon_column] <= 180)
        geo_df = geo_df[lat_valid & lon_valid]
        
        if len(geo_df) == 0:
            raise ValueError("No valid coordinates in expected ranges")
        
        # Perform clustering if requested
        cluster_info = {}
        if cluster_stations and len(geo_df) >= n_clusters:
            cluster_info = _cluster_stations(geo_df, lat_column, lon_column, n_clusters)
        
        # Create station data
        stations = []
        for idx, row in geo_df.iterrows():
            station = {
                'id': row['station_id'],
                'lat': float(row[lat_column]),
                'lon': float(row[lon_column])
            }
            
            # Add value if available
            if value_column and value_column in row:
                station[value_column] = float(row[value_column]) if pd.notnull(row[value_column]) else None
            
            # Add cluster info if clustering was performed
            if cluster_info and 'clusters' in cluster_info:
                cluster_idx = cluster_info['clusters'].get(idx, 0)
                station['cluster'] = int(cluster_idx)
            
            # Detect anomaly flag if available (from previous anomaly detection)
            station['anomaly'] = False  # Default, can be updated by anomaly detection
            
            stations.append(station)
        
        # Generate GeoJSON
        geojson = _create_geojson(stations, value_column)
        
        # Calculate bounds
        bounds = {
            'north': float(geo_df[lat_column].max()),
            'south': float(geo_df[lat_column].min()),
            'east': float(geo_df[lon_column].max()),
            'west': float(geo_df[lon_column].min())
        }
        
        # Calculate center
        center = {
            'lat': float(geo_df[lat_column].mean()),
            'lon': float(geo_df[lon_column].mean())
        }
        
        # Generate heatmap data if value column exists
        heatmap_data = []
        if value_column and value_column in geo_df.columns:
            for _, row in geo_df.iterrows():
                if pd.notnull(row[value_column]):
                    heatmap_data.append([
                        float(row[lat_column]),
                        float(row[lon_column]),
                        float(row[value_column])
                    ])
        
        # Prepare result
        result = {
            'task': 'geospatial_mapping',
            'total_stations': len(stations),
            'stations': stations,
            'geojson': geojson,
            'bounds': bounds,
            'center': center,
            'heatmap_data': heatmap_data
        }
        
        # Add clustering results
        if cluster_info:
            result['clustering'] = cluster_info
        
        return result
        
    except Exception as e:
        return {
            'task': 'geospatial_mapping',
            'error': str(e),
            'success': False
        }

def _cluster_stations(geo_df, lat_column, lon_column, n_clusters):
    """Perform K-means clustering on station locations"""
    
    try:
        # Prepare coordinate data
        coords = geo_df[[lat_column, lon_column]].values
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Create cluster mapping
        cluster_mapping = {}
        for idx, cluster in enumerate(cluster_labels):
            original_idx = geo_df.index[idx]
            cluster_mapping[original_idx] = cluster
        
        # Calculate cluster centers and statistics
        cluster_centers = []
        cluster_stats = []
        
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_coords = coords[cluster_mask]
            
            center = {
                'lat': float(np.mean(cluster_coords[:, 0])),
                'lon': float(np.mean(cluster_coords[:, 1])),
                'cluster_id': i
            }
            cluster_centers.append(center)
            
            stats = {
                'cluster_id': i,
                'station_count': int(np.sum(cluster_mask)),
                'center_lat': center['lat'],
                'center_lon': center['lon']
            }
            cluster_stats.append(stats)
        
        return {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'clusters': cluster_mapping,
            'cluster_centers': cluster_centers,
            'cluster_stats': cluster_stats
        }
        
    except Exception as e:
        return {'error': f'Clustering failed: {str(e)}'}

def _create_geojson(stations, value_column):
    """Create GeoJSON FeatureCollection from station data"""
    
    features = []
    
    for station in stations:
        # Create feature geometry
        geometry = {
            'type': 'Point',
            'coordinates': [station['lon'], station['lat']]
        }
        
        # Create feature properties
        properties = {
            'id': station['id'],
            'lat': station['lat'],
            'lon': station['lon']
        }
        
        # Add value if available
        if value_column and value_column in station:
            properties[value_column] = station[value_column]
        
        # Add other properties
        for key, value in station.items():
            if key not in ['id', 'lat', 'lon', value_column]:
                properties[key] = value
        
        # Create feature
        feature = {
            'type': 'Feature',
            'geometry': geometry,
            'properties': properties
        }
        
        features.append(feature)
    
    # Create FeatureCollection
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    return geojson

def find_similar_stations(df, lat_column, lon_column, value_column, target_station_id, k=5):
    """
    Find stations with similar patterns to a target station.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    lat_column : str
        Name of latitude column
    lon_column : str  
        Name of longitude column
    value_column : str
        Name of value column for similarity comparison
    target_station_id : str
        ID of target station
    k : int
        Number of similar stations to return
    
    Returns:
    --------
    dict : Similar stations with distances and similarity scores
    """
    
    try:
        # Find target station
        if 'station_id' in df.columns:
            target_row = df[df['station_id'] == target_station_id]
        else:
            # Assume first row if no station_id column
            target_row = df.iloc[[0]]
        
        if len(target_row) == 0:
            raise ValueError(f"Target station '{target_station_id}' not found")
        
        target_lat = target_row[lat_column].iloc[0]
        target_lon = target_row[lon_column].iloc[0]
        target_value = target_row[value_column].iloc[0] if pd.notnull(target_row[value_column].iloc[0]) else 0
        
        # Calculate similarities
        similarities = []
        
        for idx, row in df.iterrows():
            if 'station_id' in df.columns and row['station_id'] == target_station_id:
                continue  # Skip target station itself
            
            # Geographic distance (haversine)
            coords1 = np.radians([[target_lat, target_lon]])
            coords2 = np.radians([[row[lat_column], row[lon_column]]])
            distance_km = haversine_distances(coords1, coords2)[0, 0] * 6371000 / 1000  # Convert to km
            
            # Value similarity (normalized)
            current_value = row[value_column] if pd.notnull(row[value_column]) else 0
            value_diff = abs(target_value - current_value)
            value_similarity = 1 / (1 + value_diff)  # Higher similarity for smaller differences
            
            # Combined similarity score (weighted)
            geographic_weight = 0.3
            value_weight = 0.7
            
            # Normalize geographic distance (inverse relationship)
            max_distance = 1000  # Assume max relevant distance is 1000km
            geographic_similarity = max(0, 1 - distance_km / max_distance)
            
            combined_score = (geographic_weight * geographic_similarity + 
                            value_weight * value_similarity)
            
            similarity_info = {
                'station_id': row.get('station_id', f'ST{idx}'),
                'lat': float(row[lat_column]),
                'lon': float(row[lon_column]),
                'value': float(current_value),
                'distance_km': float(distance_km),
                'value_similarity': float(value_similarity),
                'combined_score': float(combined_score)
            }
            
            similarities.append(similarity_info)
        
        # Sort by combined score and return top k
        similarities.sort(key=lambda x: x['combined_score'], reverse=True)
        top_similar = similarities[:k]
        
        result = {
            'task': 'station_similarity',
            'target_station': {
                'id': target_station_id,
                'lat': float(target_lat),
                'lon': float(target_lon),
                'value': float(target_value)
            },
            'similar_stations': top_similar,
            'k': k
        }
        
        return result
        
    except Exception as e:
        return {
            'task': 'station_similarity',
            'error': str(e),
            'success': False
        }

def generate_station_heatmap_data(df, lat_column, lon_column, value_column, grid_size=20):
    """
    Generate heatmap intensity data for station visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    lat_column : str
        Name of latitude column
    lon_column : str
        Name of longitude column
    value_column : str
        Name of value column for intensity
    grid_size : int
        Grid resolution for heatmap
    
    Returns:
    --------
    dict : Heatmap data with grid coordinates and intensities
    """
    
    try:
        # Prepare data
        valid_data = df[[lat_column, lon_column, value_column]].dropna()
        
        if len(valid_data) == 0:
            raise ValueError("No valid data for heatmap")
        
        # Define bounds
        lat_min, lat_max = valid_data[lat_column].min(), valid_data[lat_column].max()
        lon_min, lon_max = valid_data[lon_column].min(), valid_data[lon_column].max()
        
        # Create grid
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        # Calculate intensities
        heatmap_points = []
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                # Calculate weighted intensity based on nearby stations
                total_weight = 0
                weighted_value = 0
                
                for _, row in valid_data.iterrows():
                    # Distance to grid point
                    coords1 = np.radians([[lat, lon]])
                    coords2 = np.radians([[row[lat_column], row[lon_column]]])
                    distance_km = haversine_distances(coords1, coords2)[0, 0] * 6371000 / 1000
                    
                    # Weight based on inverse distance
                    if distance_km == 0:
                        weight = 1e6  # Very close
                    else:
                        weight = 1 / (1 + distance_km)
                    
                    total_weight += weight
                    weighted_value += weight * row[value_column]
                
                if total_weight > 0:
                    intensity = weighted_value / total_weight
                    heatmap_points.append([lat, lon, intensity])
        
        result = {
            'task': 'heatmap_generation',
            'grid_size': grid_size,
            'bounds': {
                'north': float(lat_max),
                'south': float(lat_min),
                'east': float(lon_max),
                'west': float(lon_min)
            },
            'heatmap_data': heatmap_points
        }
        
        return result
        
    except Exception as e:
        return {
            'task': 'heatmap_generation',
            'error': str(e),
            'success': False
        }
