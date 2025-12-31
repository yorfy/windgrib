# WindGrib Usage Modes

This document details the different specific usage modes of the WindGrib library, covering common and advanced usage scenarios.

## Table of Contents

1. [Download Mode](#download-mode)
2. [Reading Mode](#reading-mode)
3. [Conversion Mode](#conversion-mode)
4. [Analysis Mode](#analysis-mode)
5. [Comparison Mode](#comparison-mode)
6. [Batch Mode](#batch-mode)
7. [Interactive Mode](#interactive-mode)
8. [Development Mode](#development-mode)

## Download Mode

### Description
The download mode is the most fundamental usage mode. It allows downloading GRIB data from remote sources (NOAA, ECMWF) and storing it locally.

### Use Cases
- Initial data download
- Regular data updates
- Creating a local cache of meteorological data

### Examples

```python
# Simple download
grib = Grib(model='gfswave')
grib.download()

# Download with options
grib = Grib(
    time='2023-12-25 12:00',
    model='ecmwf_ifs',
    data_path='./data/meteo'
)
grib.download(use_cache=False)

# Selective download
grib = Grib(model='ecmwf_ifs')
grib.download()
wind_subset = grib._subsets['wind']
wind_subset.download()  # Download only the wind subset
```

### Best Practices
- Always use `use_cache=True` to avoid unnecessary downloads
- Specify custom paths to organize your data
- Limit the number of workers for limited network connections

## Reading Mode

### Description
The reading mode allows loading and accessing already downloaded GRIB data without performing new downloads.

### Use Cases
- Analysis of existing data
- Batch processing of historical data
- Quick data access for visualization

### Examples

```python
# Simple reading
grib = Grib(model='gfswave')
wind_data = grib['wind']  # Automatically loads if needed

# Reading with cache management
grib = Grib(model='ecmwf_ifs')
# Check if data exists before loading
if wind_subset.grib_file.exists():
    ds = wind_subset.load_dataset()
    print(f"Data loaded: {list(ds.data_vars)}")

# Selective reading
grib = Grib(model='ecmwf_ifs')
land_data = grib['land']  # Load only land data
```

### Best Practices
- Always check file existence before loading
- Use bracket notation `grib['subset']` for quick access
- Handle loading errors gracefully

## Conversion Mode

### Description
The conversion mode allows transforming GRIB files into other formats more suitable for analysis, such as NetCDF.

### Use Cases
- Conversion for use with other tools
- Performance optimization for reading
- Archiving data in a standard format

### Examples

```python
# Complete conversion
grib = Grib(model='gfswave')
grib.download()
grib.to_nc()  # Convert all subsets

# Selective conversion
grib = Grib(model='ecmwf_ifs')
grib.download()
wind_subset = grib._subsets['wind']
wind_subset.to_nc()  # Convert only the wind subset

# Conversion with custom options
subset = grib._subsets['wind']
ds = subset.load_grib_file()

# Custom encoding
custom_encoding = {
    var: {
        'dtype': 'float32',
        'scale_factor': 0.01,
        'zlib': True,
        'complevel': 5
    }
    for var in ds.data_vars
}

ds.to_netcdf(subset.nc_file, encoding=custom_encoding)
```

### Best Practices
- Always convert after download to avoid data loss
- Use optimized encodings to reduce file size
- Keep original GRIB files as backup

## Analysis Mode

### Description
The analysis mode allows performing calculations and processing on loaded meteorological data.

### Use Cases
- Calculating derived parameters (wind speed, direction)
- Statistical analysis of data
- Spatial and temporal processing

### Examples

```python
# Basic calculations
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Calculate wind speed
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)

# Calculate wind direction
wind_direction = np.arctan2(wind_data.v, wind_data.u) * (180 / np.pi) % 360

# Statistics
stats = {
    'mean': wind_speed.mean(),
    'max': wind_speed.max(),
    'min': wind_speed.min(),
    'std': wind_speed.std()
}

# Temporal analysis
# Calculate trend
time_series = wind_speed.mean(dim=['latitude', 'longitude'])
tendency = np.gradient(time_series.values)

# Spatial analysis
# Calculate spatial gradient
lat_grad, lon_grad = np.gradient(wind_speed.isel(step=0).values)
```

### Best Practices
- Use vectorized operations from xarray/numpy for performance
- Work with chunks for large datasets
- Always validate calculation results

## Comparison Mode

### Description
The comparison mode allows comparing data between different models or different periods.

### Use Cases
- Comparison between models (ECMWF vs GFS)
- Analysis of differences between forecasts
- Validation of weather models

### Examples

```python
# Comparison between models
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_gfs = Grib(model='gfswave')

grib_ecmwf.download()
grib_gfs.download()

# Load data
 ecmwf_wind = grib_ecmwf['wind']
 gfs_wind = grib_gfs['wind']

# Calculate speeds
 ecmwf_speed = np.sqrt(ecmwf_wind.u**2 + ecmwf_wind.v**2)
 gfs_speed = np.sqrt(gfs_wind.u**2 + gfs_wind.v**2)

# Direct comparison
difference = ecmwf_speed - gfs_speed
relative_difference = difference / ecmwf_speed * 100

# Comparison statistics
comparison_stats = {
    'mean_abs_diff': np.abs(difference).mean(),
    'max_abs_diff': np.abs(difference).max(),
    'mean_rel_diff': np.abs(relative_difference).mean(),
    'correlation': np.corrcoef(
        ecmwf_speed.values.flatten(),
        gfs_speed.values.flatten()
    )[0, 1]
}

# Temporal comparison
# Align time steps
 ecmwf_times = pd.to_datetime(ecmwf_wind.valid_time.values)
 gfs_times = pd.to_datetime(gfs_wind.valid_time.values)

# Find common times
common_times = ecmwf_times.intersection(gfs_times)

# Comparison for common times
if len(common_times) > 0:
    ecmwf_common = ecmwf_speed.sel(valid_time=common_times)
    gfs_common = gfs_speed.sel(valid_time=common_times)
    
    time_series_comparison = {
        'ecmwf': ecmwf_common.mean(dim=['latitude', 'longitude']),
        'gfs': gfs_common.mean(dim=['latitude', 'longitude'])
    }
```

### Best Practices
- Always align grids and time steps before comparison
- Use normalized metrics for comparisons
- Visualize differences for better understanding

## Batch Mode

### Description
The batch mode allows performing operations on multiple datasets or periods in an automated way.

### Use Cases
- Downloading historical data
- Batch processing of multiple models
- Analysis of long time series

### Examples

```python
# Batch download for multiple dates
dates = [
    '2023-12-20', '2023-12-21', '2023-12-22',
    '2023-12-23', '2023-12-24', '2023-12-25'
]

for date in dates:
    try:
        grib = Grib(time=date, model='gfswave')
        grib.download()
        grib.to_nc()
        print(f"✅ Processed: {date}")
    except Exception as e:
        print(f"❌ Failed for {date}: {e}")

# Batch processing for multiple models
models = ['gfswave', 'ecmwf_ifs']

for model in models:
    try:
        grib = Grib(model=model)
        grib.download()
        
        # Process each subset
        for subset_name in grib.subset_names:
            subset = grib._subsets[subset_name]
            ds = subset.ds
            
            # Perform calculations
            if 'u' in ds.data_vars and 'v' in ds.data_vars:
                wind_speed = np.sqrt(ds.u**2 + ds.v**2)
                mean_speed = wind_speed.mean()
                print(f"{model} - {subset_name}: {mean_speed.values:.2f} m/s")
                
    except Exception as e:
        print(f"Failed for model {model}: {e}")

# Batch analysis with parallelism
from concurrent.futures import ThreadPoolExecutor

def process_date(date):
    try:
        grib = Grib(time=date, model='gfswave')
        grib.download()
        wind_data = grib['wind']
        mean_speed = np.sqrt(wind_data.u**2 + wind_data.v**2).mean()
        return date, mean_speed.values
    except Exception as e:
        return date, None

# Parallel execution
dates_to_process = [f'2023-12-{day:02d}' for day in range(20, 26)]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_date, dates_to_process))

# Display results
for date, speed in results:
    if speed is not None:
        print(f"{date}: {speed:.2f} m/s")
    else:
        print(f"{date}: Failed")
```

### Best Practices
- Limit parallelism based on your resources
- Handle errors individually for each task
- Save intermediate results

## Interactive Mode

### Description
The interactive mode is designed for use in interactive environments like Jupyter Notebooks, where the user can explore and visualize data iteratively.

### Use Cases
- Interactive data exploration
- Ad-hoc visualization and analysis
- Rapid prototyping of algorithms

### Examples

```python
# Interactive exploration in Jupyter
%matplotlib inline
import matplotlib.pyplot as plt
from windgrib import Grib
import numpy as np

# Load data
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Explore variables
print("Available variables:", list(wind_data.data_vars))
print("Dimensions:", dict(wind_data.sizes))
print("Coordinates:", list(wind_data.coords))

# Interactive visualization
def plot_wind_step(step=0):
    """Function to visualize a specific time step."""
    wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Wind speed
    wind_speed.isel(step=step).plot(ax=axes[0], cmap='viridis')
    axes[0].set_title(f'Wind Speed - Step {step}')
    
    # Wind direction
    wind_direction = np.arctan2(wind_data.v, wind_data.u).isel(step=step) * (180 / np.pi) % 360
    wind_direction.plot(ax=axes[1], cmap='hsv')
    axes[1].set_title(f'Wind Direction - Step {step}')
    
    plt.tight_layout()
    plt.show()

# Interactive usage
plot_wind_step(0)
plot_wind_step(5)
plot_wind_step(-1)

# Interactive analysis
def analyze_region(lat_min=30, lat_max=50, lon_min=-20, lon_max=20):
    """Analysis of a specific region."""
    region = wind_data.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )
    
    wind_speed = np.sqrt(region.u**2 + region.v**2)
    
    # Statistics
    stats = {
        'mean': wind_speed.mean().values,
        'max': wind_speed.max().values,
        'min': wind_speed.min().values,
        'std': wind_speed.std().values
    }
    
    # Visualization
    plt.figure(figsize=(10, 6))
    wind_speed.mean(dim='step').plot(cmap='plasma')
    plt.title(f'Mean Wind Speed\nLat: {lat_min}-{lat_max}°, Lon: {lon_min}-{lon_max}°')
    plt.colorbar(label='Speed (m/s)')
    plt.show()
    
    return stats

# Interactive usage
stats_europe = analyze_region(35, 60, -15, 30)
stats_atlantic = analyze_region(20, 40, -60, -20)

# Interactive comparison
def compare_steps(step1=0, step2=1):
    """Comparison between two time steps."""
    wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
    
    speed1 = wind_speed.isel(step=step1)
    speed2 = wind_speed.isel(step=step2)
    
    difference = speed2 - speed1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    speed1.plot(ax=axes[0], cmap='viridis')
    axes[0].set_title(f'Step {step1}')
    
    speed2.plot(ax=axes[1], cmap='viridis')
    axes[1].set_title(f'Step {step2}')
    
    difference.plot(ax=axes[2], cmap='coolwarm', vmin=-5, vmax=5)
    axes[2].set_title(f'Difference (Step {step2} - Step {step1})')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean difference: {difference.mean().values:.3f} m/s")
    print(f"Max difference: {difference.max().values:.3f} m/s")
    print(f"Min difference: {difference.min().values:.3f} m/s")

# Interactive usage
compare_steps(0, 10)
compare_steps(5, 15)
```

### Best Practices
- Use reusable functions for exploration
- Document your interactive analyses
- Save interesting results for future reference

## Development Mode

### Description
The development mode is used to extend WindGrib's functionality or to integrate the library into other systems.

### Use Cases
- Adding new weather models
- Extending existing functionality
- Integration with other libraries

### Examples

```python
# Adding a new model
from windgrib.grib import MODELS, Grib

# Define a new model
MODELS['my_new_model'] = {
    'product': 'my_product',
    'url': 'https://my-weather-source.com/',
    'key': '{date}/{h:02d}/my_model/',
    'subsets': {
        'temperature': {'variable': ['TMP']},
        'pressure': {'variable': ['PRMSL']},
        'humidity': {'variable': ['RH']}
    }
}

# Use the new model
try:
    grib = Grib(model='my_new_model')
    grib.download()
    temp_data = grib['temperature']
    print(f"New model working: {list(temp_data.data_vars)}")
except Exception as e:
    print(f"Error with new model: {e}")

# Extending the GribSubset class
from windgrib.grib import GribSubset

class MyExtendedGribSubset(GribSubset):
    def calculate_additional_parameter(self):
        """Calculate an additional meteorological parameter."""
        ds = self.ds
        
        if 'u' in ds.data_vars and 'v' in ds.data_vars:
            # Calculate divergence
            wind_speed = np.sqrt(ds.u**2 + ds.v**2)
            
            # Simplified divergence calculation
            # (In reality, appropriate finite differences should be used)
            divergence = np.gradient(wind_speed.values)
            
            return divergence
        
        return None

# Using the extended class
grib = Grib(model='gfswave')
grib.download()

# Replace existing subset with our extended version
wind_subset = grib._subsets['wind']
extended_subset = MyExtendedGribSubset(
    name='wind',
    config=wind_subset.config,
    grib_instance=grib
)

# Use new functionality
divergence = extended_subset.calculate_additional_parameter()
if divergence is not None:
    print(f"Divergence calculated: shape={divergence[0].shape}")

# Integration with other libraries
def integrate_with_pandas(grib_instance):
    """Integration with pandas for time series analysis."""
    wind_data = grib_instance['wind']
    
    # Convert to pandas DataFrame
    wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
    
    # Spatial average for each time step
    time_series = wind_speed.mean(dim=['latitude', 'longitude'])
    
    # Convert to DataFrame
    df = time_series.to_dataframe(name='wind_speed')
    df['model'] = grib_instance.model['name']
    
    return df

# Usage
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_ecmwf.download()
df_ecmwf = integrate_with_pandas(grib_ecmwf)

grib_gfs = Grib(model='gfswave')
grib_gfs.download()
df_gfs = integrate_with_pandas(grib_gfs)

# Combine results
combined_df = pd.concat([df_ecmwf, df_gfs])
print(combined_df.head())

# Comparative analysis
comparison = combined_df.pivot(columns='model', values='wind_speed')
print(comparison.describe())
```

### Best Practices
- Follow existing coding conventions
- Document your extensions well
- Test new features thoroughly
- Maintain compatibility with existing versions

## Conclusion

WindGrib offers great flexibility in its usage modes, allowing it to meet various needs from simple data download to complex meteorological analysis. By understanding these different modes, you can get the most out of the library for your specific applications.

For more information on each mode, consult the specific examples and detailed technical documentation.