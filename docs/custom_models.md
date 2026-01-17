# Creating Custom Models with WindGrib

This guide explains how to extend WindGrib to support new data types by creating custom models.

## Table of Contents

1. [Introduction](#introduction)
2. [Model Structure](#model-structure)
3. [Adding a Basic Model](#adding-a-basic-model)
4. [Advanced Model with Parameters](#advanced-model-with-parameters)
5. [Extending Classes](#extending-classes)
6. [Complete Examples](#complete-examples)
7. [Best Practices](#best-practices)

## Introduction

WindGrib is designed to be extensible, allowing users to add custom weather models to download and process different types of data. This guide shows you how to:

- Add new data models
- Configure custom subsets
- Extend existing functionality
- Integrate with other libraries

## Model Structure

A WindGrib model is defined by a dictionary with the following structure:

```python
MODELS['my_model'] = {
    'product': 'product_id',            # Product identifier
    'url': 'https://source.com/',       # Base URL
    'key': '{date}/{h:02d}/path/',     # Relative path (with formatting)
    'idx': '.idx',                     # Index file extension (optional)
    'subsets': {                       # Data subsets
        'subset_name': {
            'variable': ['VAR1', 'VAR2'],  # Variables to download
            'level': 'level',             # Atmospheric level (optional)
            'step': 0                      # Specific time step (optional)
        }
    },
    'var_mapping': {                   # Variable name mapping (optional)
        'old_name': 'new_name'
    },
    'filter_key': 'shortName'          # Filter key for cfgrib (optional)
}
```

## Adding a Basic Model

### Example: Temperature Model

Here's how to add a model for downloading temperature data:

```python
from windgrib.grib import MODELS

# Add a model for GFS temperature data
MODELS['gfs_temp'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {
            'variable': ['TMP']  # Temperature
        },
        'humidity': {
            'variable': ['RH']   # Relative humidity
        },
        'pressure': {
            'variable': ['PRMSL']  # Sea level pressure
        }
    }
}

# Use the new model
gb = Grib(model='gfs_temp')
gb.download()

# Access the data
temp_data = gb['temperature'].ds
print(f"Mean temperature: {temp_data.TMP.mean().values:.2f} K")
```

### Explanation

1. **Model Configuration** :
   - `product`: Identifies the GFS global product at 0.25° resolution
   - `url`: Base URL for NOAA GFS data
   - `key`: Relative path with date/time formatting

2. **Subsets** :
   - `temperature`: Downloads TMP variable (temperature in Kelvin)
   - `humidity`: Downloads RH variable (relative humidity in %)
   - `pressure`: Downloads PRMSL variable (pressure in Pascals)

3. **Usage** :
   - The model is immediately available after adding
   - Data is downloaded and accessible like standard models

## Advanced Model with Parameters

### Example: Model with Mapping and Filtering

```python
# Advanced model with variable mapping and filtering
MODELS['advanced_model'] = {
    'product': 'my_product',
    'url': 'https://weather-source.example.com/',
    'key': '{date}/{h:02d}/model/',
    'idx': '.idx',  # Custom index extension
    'subsets': {
        'surface_temperature': {
            'variable': ['TMP'],
            'level': 'surface'  # Specific level
        },
        'temperature_850hPa': {
            'variable': ['TMP'],
            'level': '850'      # 850 hPa level
        },
        'geopotential': {
            'variable': ['HGT'],
            'level': '500'      # 500 hPa level
        }
    },
    'var_mapping': {
        'TMP': 'temperature',   # Rename TMP to temperature
        'HGT': 'geopotential'   # Rename HGT to geopotential
    },
    'filter_key': 'shortName'  # Filter key for cfgrib
}

# Using the advanced model
gb = Grib(model='advanced_model')
gb.download()

# Variables are renamed according to var_mapping
data = gb['surface_temperature'].ds
print(f"Available variables: {list(data.data_vars)}")
# Shows: ['temperature'] instead of ['TMP']
```

### Advanced Parameters

1. **`var_mapping`** : Allows renaming variables for clarity
2. **`filter_key`** : Specifies the key to use for filtering variables with cfgrib
3. **`level`** : Allows specifying atmospheric levels
4. **`step`** : Allows limiting to specific time steps

## Extending Classes

### Extending GribSubset for Custom Functionality

```python
from windgrib.grib import GribSubset
import numpy as np


class MyExtendedGribSubset(GribSubset):
   """Extended class with additional functionality."""

   def calculate_thermal_indices(self):
      """Calculate thermal indices from the data."""
      ds = self.ds

      if ds is None:
         return None

      results = {}

      # Example: Calculate simplified heat index
      if 'temperature' in ds.data_vars and 'humidity' in ds.data_vars:
         temp = ds.temperature
         rh = ds.humidity
         heat_index = temp + 0.01 * rh  # Simplified formula
         results['heat_index'] = heat_index

      # Calculate extremes
      for var_name in ds.data_vars:
         results[f'{var_name}_max'] = ds[var_name].max()
         results[f'{var_name}_min'] = ds[var_name].min()

      return results

   def analyze_trends(self):
      """Analyze temporal trends."""
      ds = self.ds

      if ds is None or 'step' not in ds.dims:
         return None

      results = {}

      for var_name in ds.data_vars:
         # Calculate linear trend
         mean_series = ds[var_name].mean(dim=['latitude', 'longitude'])

         # Linear regression
         x = np.arange(len(mean_series))
         y = mean_series.values
         A = np.vstack([x, np.ones(len(x))]).T
         m, c = np.linalg.lstsq(A, y, rcond=None)[0]

         results[var_name] = {
            'slope': m,
            'intercept': c,
            'trend': 'increasing' if m > 0 else 'decreasing' if m < 0 else 'stable'
         }

      return results


# Using the extended class
gb = Grib(model='gfswave')
gb.download()

# Replace subset with our extended version
wind_subset = gb['wind']
extended_subset = MyExtendedGribSubset(
   name='wind',
   config=wind_subset.filter_keys,
   grib_instance=gb
)

# Use new functionality
indices = extended_subset.calculate_thermal_indices()
trends = extended_subset.analyze_trends()

print(f"Calculated indices: {list(indices.keys()) if indices else 'None'}")
print(f"Trends: {trends}")
```

### Custom Features

1. **Advanced Meteorological Calculations** : Heat indices, wind chill indices, etc.
2. **Trend Analysis** : Detection of temporal and spatial trends
3. **Domain-Specific Processing** : Tailored to your analysis needs
4. **Easy Integration** : Compatible with WindGrib's existing API

## Complete Examples

### Example 1: Temperature Data Model (from examples/gfs_temperature_example.py)

```python
# Simplified GFS atmospheric model for temperature data
MODELS['gfs_atmos'] = {
    'product': 'pgrb2.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {
            'variable': ['TMP']  # Temperature variable
        }
    }
}

# Usage
print("=== WindGrib Example: GFS Atmospheric Data ===\n")

# 1. Initialization
print("1. Initializing GFS atmospheric model...")
gb = Grib(model='gfs_atmos')

print("2. Downloading data...")
try:
    gb.download()  # Use current API
    print("[OK] Download completed")
except Exception as e:
    print(f"[ERROR] Download error: {e}")
    exit()

# 2. Data analysis
print("\n3. Analyzing temperature data...")

try:
    temp_data = gb['temperature']
    print(f"Available variables: {list(temp_data.data_vars)}")
    
    # Convert Kelvin to Celsius
    temp_celsius = temp_data.TMP - 273.15
    
    print(f"Average temperature: {temp_celsius.mean().values:.2f}°C")
    print(f"Minimum temperature: {temp_celsius.min().values:.2f}°C")
    print(f"Maximum temperature: {temp_celsius.max().values:.2f}°C")
    
except Exception as e:
    print(f"[ERROR] Analysis error: {e}")

print("[OK] Example completed")
```

### Example 2: Precipitation Data Model

```python
# Add a model for precipitation data
MODELS['gfs_precip'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'precipitation': {
            'variable': ['APCP']  # Accumulated precipitation
        },
        'precip_rate': {
            'variable': ['PRATE']  # Precipitation rate
        }
    }
}

# Usage
gb = Grib(model='gfs_precip')
gb.download()

precip_data = gb['precipitation'].ds
print(f"Total precipitation: {precip_data.APCP.sum().values:.2f} kg/m²")
```

### Example 2: Multi-Level Model

```python
# Model with multiple atmospheric levels
MODELS['gfs_levels'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temp_surface': {'variable': ['TMP'], 'level': 'surface'},
        'temp_850': {'variable': ['TMP'], 'level': '850'},
        'temp_500': {'variable': ['TMP'], 'level': '500'},
        'hgt_500': {'variable': ['HGT'], 'level': '500'},
        'uwnd_250': {'variable': ['UGRD'], 'level': '250'},
        'vwnd_250': {'variable': ['VGRD'], 'level': '250'}
    }
}

# Usage to calculate wind at 250 hPa
gb = Grib(model='gfs_levels')
gb.download()

u250 = gb['uwnd_250'].ds
v250 = gb['vwnd_250'].ds
wind_250 = np.sqrt(u250.UGRD**2 + v250.VGRD**2)
print(f"Mean wind speed at 250 hPa: {wind_250.mean().values:.2f} m/s")
```

### Example 3: Model Comparison with Visualization (from examples/ecmf_gfs_wind_comparison.py)

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from windgrib import Grib


def ecmf_gfs_wind_speed_comparison():
   # Comparison of wind speed forecast from ECMWF and GFS

   # ECMWF download
   print("=== ECMWF download ===\n")
   grib_ecmwf = Grib(model='ecmwf_ifs')
   grib_ecmwf.download()
   grib_ecmwf.to_netcdf()

   # GFS download
   print("\n=== GFS download ===\n")
   grib_gfs = Grib()
   grib_gfs.download()
   grib_gfs.to_netcdf()

   # Get datasets
   ecmwf_wind_ds = grib_ecmwf['wind'].ds
   ecmwf_land_ds = grib_ecmwf['land'].ds
   gfs_ds = grib_gfs['wind'].ds

   # Convert GFS longitude from 0-360 to -180-180
   if 'longitude' in gfs_ds.coords:
      gfs_ds = gfs_ds.assign_coords(longitude=((gfs_ds.longitude + 180) % 360) - 180)
      gfs_ds = gfs_ds.sortby('longitude')

   print(f"\nECMWF wind variables: {list(ecmwf_wind_ds.data_vars)}")
   print(f"ECMWF wind dimensions: {dict(ecmwf_wind_ds.sizes)}")
   if ecmwf_land_ds:
      print(f"ECMWF land variables: {list(ecmwf_land_ds.data_vars)}")
   print(f"\nGFS loaded variables: {list(gfs_ds.data_vars)}")
   print(f"GFS dimensions: {dict(gfs_ds.sizes)}")

   # Wind speed comparison
   print("\n=== Wind Speed Comparison ===\n")

   # Find common valid times
   ecmwf_valid_times = pd.to_datetime(ecmwf_wind_ds.valid_time.values)
   gfs_valid_times = pd.to_datetime(gfs_ds.valid_time.values)
   common_times = ecmwf_valid_times.intersection(gfs_valid_times)

   if len(common_times) == 0:
      print("No common valid times found between ECMWF and GFS")
      current_time = pd.Timestamp.now()
      ecmwf_closest_idx = np.abs(ecmwf_valid_times - current_time).argmin()
      gfs_closest_idx = np.abs(gfs_valid_times - current_time).argmin()
      ecmwf_time = ecmwf_valid_times[ecmwf_closest_idx]
      gfs_time = gfs_valid_times[gfs_closest_idx]
      print(f"Using closest times to now ({current_time}):")
      print(f"ECMWF: {ecmwf_time} (step={ecmwf_closest_idx})")
      print(f"GFS: {gfs_time} (step={gfs_closest_idx})")
      ecmwf_step = ecmwf_closest_idx
      gfs_step = gfs_closest_idx
   else:
      current_time = pd.Timestamp.now()
      closest_common_idx = np.abs(common_times - current_time).argmin()
      closest_common_time = common_times[closest_common_idx]
      ecmwf_step = list(ecmwf_valid_times).index(closest_common_time)
      gfs_step = list(gfs_valid_times).index(closest_common_time)
      print(f"Using common time closest to now ({current_time}):")
      print(f"Common time: {closest_common_time}")
      print(f"ECMWF step: {ecmwf_step}, GFS step: {gfs_step}")

   # Calculate wind speed in m/s then convert to knots
   ecmwf_speed = (ecmwf_wind_ds.u ** 2 + ecmwf_wind_ds.v ** 2) ** 0.5
   ecmwf_speed_knots = ecmwf_speed * 1.94384  # Convert m/s to knots
   ecmwf_speed_knots.attrs['units'] = 'knots'
   ecmwf_speed_knots.attrs['long_name'] = 'Wind Speed'

   if 'v' in gfs_ds.data_vars:
      gfs_speed = (gfs_ds.u ** 2 + gfs_ds.v ** 2) ** 0.5
   else:
      print("Warning: GFS missing v component, using only u component")
      gfs_speed = abs(gfs_ds.u)
   gfs_speed_knots = gfs_speed * 1.94384  # Convert m/s to knots
   gfs_speed_knots.attrs['units'] = 'knots'
   gfs_speed_knots.attrs['long_name'] = 'Wind Speed'

   # Apply ocean mask to ECMWF wind speed
   if ecmwf_land_ds and 'lsm' in ecmwf_land_ds.data_vars:
      lsm = ecmwf_land_ds.lsm
      ocean_mask = lsm < 0.5
      ecmwf_speed_masked = ecmwf_speed_knots.where(ocean_mask)
      print("Applied ocean mask to ECMWF wind speed")
   else:
      ecmwf_speed_masked = ecmwf_speed_knots
      print("No ocean mask applied - LSM not available")

   # Plot comparison
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   ecmwf_speed_masked.isel(step=ecmwf_step).plot(ax=ax1, cmap='viridis')
   ax1.set_title(f'ECMWF Wind Speed (Ocean Only)\n{ecmwf_valid_times[ecmwf_step]}')

   gfs_speed_knots.isel(step=gfs_step).plot(ax=ax2, cmap='viridis')
   ax2.set_title(f'GFS Wind Speed\n{gfs_valid_times[gfs_step]}')
   plt.tight_layout()
   plt.savefig('docs/images/ecmwf_gfs_comparison.png', dpi=300, bbox_inches='tight')
   plt.show()

   print("\nComparison plot completed with ocean masking! (Wind speeds in knots)")
   print("Plot saved to: docs/images/ecmwf_gfs_comparison.png")


if __name__ == '__main__':
   ecmf_gfs_wind_speed_comparison()
```

**Expected Output:**

The example will generate a comparison plot showing ECMWF and GFS wind speed data side by side. When executed with proper matplotlib installation, it will:

1. Download data from both ECMWF and GFS models
2. Calculate wind speeds and apply ocean masking
3. Generate a side-by-side comparison plot
4. Save the visualization to `docs/images/ecmwf_gfs_comparison.png`

**Visualization Description:**
- Left panel: ECMWF wind speed data (ocean areas only)
- Right panel: GFS wind speed data (full coverage)
- Color scale: Wind speed in knots (using viridis colormap)
- Titles: Show model name and timestamp
- Both panels use the same color scale for direct comparison

**Note:** The actual image will be generated when you run this example locally with matplotlib installed. The plot provides visual comparison of wind speed patterns between the two major weather models.

### Example 4: Model with Custom Processing

```python
# Custom class for thermal wind calculation
class ThermalWindSubset(GribSubset):
    def calculate_thermal_wind(self):
        """Calculate thermal wind between two levels."""
        ds = self.ds
        
        if ds is None:
            return None
        
        # Requires U and V components at two different levels
        if all(var in ds.data_vars for var in ['u_850', 'u_500', 'v_850', 'v_500']):
            # Calculate thermal wind (simplified)
            u_thermal = ds.u_500 - ds.u_850
            v_thermal = ds.v_500 - ds.v_850
            
            wind_speed = np.sqrt(u_thermal**2 + v_thermal**2)
            wind_dir = np.arctan2(v_thermal, u_thermal) * (180 / np.pi) % 360
            
            return {
                'speed': wind_speed,
                'direction': wind_dir
            }
        
        return None

# Model configuration
MODELS['gfs_thermal'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'vents': {
            'variable': ['UGRD', 'VGRD'],
            'levels': ['850', '500']  # Indicates multiple levels wanted
        }
    },
    'var_mapping': {
        'UGRD_850': 'u_850',
        'VGRD_850': 'v_850',
        'UGRD_500': 'u_500',
        'VGRD_500': 'v_500'
    }
}

# Usage
thermal_wind = thermal_subset.calculate_thermal_wind()
if thermal_wind:
    print(f"Mean thermal wind: {thermal_wind['speed'].mean().values:.2f} m/s")
    print(f"Mean direction: {thermal_wind['direction'].mean().values:.1f}°")
```

## Best Practices

### 1. Model Validation

```python
# Always validate that a model exists before using it
def use_model_safely(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}")
    
    try:
        gb = Grib(model=model_name)
        gb.download()
        return gb
    except Exception as e:
        print(f"Error with {model_name} model: {e}")
        return None
```

### 2. Error Handling

```python
# Error handling for custom models
def download_model_with_retry(model_name, max_retries=3):
    """Download a model with retry logic."""

    for attempt in range(max_retries):
        try:
            gb = Grib(model=model_name)
            gb.download()

            # Verify data is valid
            for subset_name in gb:
                subset = gb[subset_name]
                if not subset.grib_file.exists():
                    raise ValueError(f"Missing file for {subset_name}")

            return grib

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise

            # Wait before retrying
            import time
            time.sleep(2 ** attempt)  # Exponential backoff

    return None
```

### 3. Model Documentation

```python
# Add documentation to your custom models
MODELS['my_model'] = {
    '__doc__': """
    Custom model for XYZ data
    
    Source: Example Weather Organization
    Resolution: 0.5° x 0.5°
    Frequency: 6 hours
    Variables: Temperature, Humidity, Pressure
    
    Reference: https://example.com/docs
    """,
    'product': 'xyz_product',
    'url': 'https://example.com/data/',
    'key': '{date}/{h:02d}/xyz/',
    'subsets': {
        # ... subset configuration
    }
}

# Access documentation
print(MODELS['my_model']['__doc__'])
```

### 4. Model Testing

```python
# Create tests for your custom models
def test_model(model_name, test_date='2023-12-25'):
   """Test that a custom model works correctly."""

   import pandas as pd

   print(f"Testing {model_name} model...")

   try:
      # Test initialization
      gb = Grib(model=model_name, time=test_date)
      print(f"✅ Initialization successful")

      # Test downloading (with minimal files for testing)
      # Note: In real tests, you might mock network calls

      # Test subset access
      for subset_name in gb:
         subset = gb[subset_name]
         print(f"  Subset {subset_name}: {subset.variables}")

      print(f"✅ Model {model_name} validated successfully")
      return True

   except Exception as e:
      print(f"❌ Test failed: {e}")
      return False


# Run tests
models_to_test = ['gfs_temp', 'advanced_model']
for model in models_to_test:
   if model in MODELS:
      test_model(model)
```

### 5. Integration with Other Libraries

```python
# Example of integration with pandas for analysis
def analyze_with_pandas(grib_instance):
    """Integrate WindGrib data with pandas for analysis."""
    
    import pandas as pd
    
    results = {}
    
    for subset_name in grib_instance:
        try:
            data = grib_instance[subset_name].ds
            
            if data is not None:
                # Calculate time averages
                for var_name in data.data_vars:
                    time_series = data[var_name].mean(dim=['latitude', 'longitude'])
                    df = time_series.to_dataframe(name=var_name)
                    df['subset'] = subset_name
                    df['model'] = grib_instance.model['name']
                    
                    results[f'{subset_name}_{var_name}'] = df
        
        except Exception as e:
            print(f"⚠️  Error with {subset_name}: {e}")
    
    # Combine all results
    if results:
        combined_df = pd.concat(results.values())
        return combined_df
    
    return None

# Usage
gb = Grib(model='gfs_temp')
gb.download()
df = analyze_with_pandas(gb)
if df is not None:
    print(f"Combined DataFrame: {df.shape}")
    print(df.head())
```

## Conclusion

This guide has shown you how to extend WindGrib to support new data types by creating custom models. Key takeaways:

1. **Model Structure** : Understand the basic structure of a WindGrib model
2. **Simple Addition** : Start with basic models before adding complexity
3. **Advanced Features** : Use variable mapping and filtering for sophisticated models
4. **Class Extension** : Create subclasses to add domain-specific functionality
5. **Best Practices** : Validate, document, and test your custom models

To go further, you can:

- Explore integration with other libraries like xarray, pandas, and matplotlib
- Create models for data sources specific to your organization
- Contribute your models to the WindGrib community
- Extend WindGrib's core functionality to support new data formats

Feel free to consult other parts of the documentation for more detailed information on advanced features and usage examples.
