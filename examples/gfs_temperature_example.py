"""
WindGrib Usage Example: GFS Atmospheric Temperature Data

This example demonstrates how to define a custom model to download
and analyze GFS atmospheric temperature data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from windgrib.grib import MODELS, Grib
import numpy as np

# Simplified GFS atmospheric model
MODELS['gfs_atmos'] = {
    'product': 'pgrb2.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {
            'variable': ['TMP']
        }
    }
}

def main():
    """Example of using the GFS atmospheric model."""
    
    print("=== WindGrib Example: GFS Atmospheric Data ===\n")
    
    # 1. Initialization
    print("1. Initializing GFS atmospheric model...")
    grib = Grib(model='gfs_atmos')
    
    print("2. Downloading data...")
    try:
        grib.download(use_cache=False)  # Force download
        print("[OK] Download completed")
    except Exception as e:
        print(f"[ERROR] Download error: {e}")
        return
    
    # 2. Data analysis
    print("\n3. Analyzing temperature data...")
    
    try:
        temp_data = grib['temperature']
        print(f"Available variables: {list(temp_data.data_vars)}")
        
        # Convert Kelvin to Celsius
        temp_celsius = temp_data.TMP - 273.15
        
        print(f"Average temperature: {temp_celsius.mean().values:.2f}°C")
        print(f"Minimum temperature: {temp_celsius.min().values:.2f}°C")
        print(f"Maximum temperature: {temp_celsius.max().values:.2f}°C")
        
    except Exception as e:
        print(f"[ERROR] Analysis error: {e}")
    
    print("[OK] Example completed")

def analyze_temperature_trends():
    """Analyze temperature trends."""
    
    print("\n=== Temperature Trend Analysis ===")
    
    try:
        grib = Grib(model='gfs_atmos')
        grib.download(use_cache=False)
        
        temp_data = grib['temperature']
        temp_celsius = temp_data.TMP - 273.15
        
        if 'step' in temp_celsius.dims and len(temp_celsius.step) > 1:
            global_mean = temp_celsius.mean(dim=['latitude', 'longitude'])
            
            print("Global average temperature evolution:")
            for i in range(min(5, len(global_mean.step))):
                temp_val = global_mean.isel(step=i).values
                print(f"  Step {i}: {temp_val:.2f}°C")
        else:
            print("Single time step - temporal analysis not possible")
            
    except Exception as e:
        print(f"Analysis error: {e}")

def compare_surface_levels():
    """Compare different temperature levels."""
    
    print("\n=== Temperature Level Comparison ===")
    print("Note: This functionality requires a model with multiple levels")
    
    # This function is disabled as the simplified model has only one level
    print("Functionality not available with current model")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Additional analyses
    try:
        analyze_temperature_trends()
        compare_surface_levels()
    except Exception as e:
        print(f"Additional analyses not available: {e}")
    
    print("\n=== Example completed ===")