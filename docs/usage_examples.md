# WindGrib Usage Examples

This section presents concrete examples showing different usage modes of WindGrib. Each example is designed to illustrate a specific aspect of the library.

## Table of Contents

1. [Basic Download and Reading](#basic-download-and-reading)
2. [Data Visualization](#data-visualization)
3. [Model Comparison](#model-comparison)
4. [GFS Atmospheric Temperature](#gfs-atmospheric-temperature)

## Basic Download and Reading

This example shows how to download GRIB data and access it in a basic way.

**Script:** [download_grib.py](../examples/download_grib.py)

**Key Features:**
- Uses caching to avoid redundant downloads
- Converts GRIB to NetCDF for faster subsequent access
- Demonstrates basic data inspection

```python
"""Minimal example of downloading and accessing data,
using cache and saving data to netcdf format to speedup further data reading"""

from windgrib import Grib

# Create a GRIB instance for the GFS Wave model
grib = Grib(model='gfswave')

# Download the data
print("Downloading GFS Wave data...")
grib.download(use_cache=True)
# use_cache=True is the default option
# But you can use use_cache=False to force downloading ignoring cache files

# Access wind data
wind_data = grib['wind']

# Display basic information
print(f"Available variables: {list(wind_data.data_vars)}")
print(f"Dimensions: {dict(wind_data.sizes)}")
print(f"Time period: {wind_data.time.values}")

# Access specific subset
print(f"U data shape: {wind_data.u.shape}")
print(f"V data shape: {wind_data.v.shape}")

# save to netcdf file to speedup further data reading
grib.to_nc()
```

## Data Visualization

This example shows how to visualize meteorological data using matplotlib.

**Script:** [data_visualization.py](../examples/data_visualization.py)

**Note:** This example demonstrates basic xarray plotting capabilities with matplotlib integration.

```python
"""Example of data visualization with matplotlib using xarray"""
from windgrib import Grib
import numpy as np
import matplotlib.pyplot as plt

    
# Load data
print("Loading data...")
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Calculate wind speed
print("Calculating wind speed...")
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)

# Plot Wind speed - First time step
wind_speed.isel(step=0).plot(cmap='viridis')
plt.tight_layout()

# Save image
plt.savefig('wind_visualization.png', dpi=300, bbox_inches='tight')
```

### Generated Visualizations

![Wind Visualization](images/wind_visualization.png)
*Wind speed visualization from GFS Wave model*

## Model Comparison

This example shows how to compare data between different weather models (ECMWF vs GFS).

**Script:** [ecmf_gfs_wind_speed_comparison.py](../examples/ecmf_gfs_wind_speed_comparison.py)

**Key Features:**
- Downloads both ECMWF and GFS wind data
- Uses ECMWF land/sea mask (`lsm`) to filter ocean-only data
- Handles coordinate system differences between models
- Finds common forecast times for fair comparison
- Converts wind speeds to knots for maritime applications

```python
"""comparison of wind speed forecast from ECMWF and GFS"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from windgrib import Grib

# ECMWF download
print("=== ECMWF download ===")
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_ecmwf.download()

# GFS download
print("\n=== GFS download ===")
grib_gfs = Grib()
grib_gfs.download()

# Get datasets
ecmwf_wind_ds = grib_ecmwf['wind']
ecmwf_land_ds = grib_ecmwf['land']
gfs_ds = grib_gfs['wind']

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
print("\n=== Wind Speed Comparison ===")

# Find common valid times
ecmwf_valid_times = pd.to_datetime(ecmwf_wind_ds.valid_time.values)
gfs_valid_times = pd.to_datetime(gfs_ds.valid_time.values)
common_times = ecmwf_valid_times.intersection(gfs_valid_times)
current_time = pd.Timestamp.utcnow().tz_localize(None)
closest_common_idx = np.abs(common_times - current_time).argmin()
closest_common_time = common_times[closest_common_idx]
ecmwf_step = list(ecmwf_valid_times).index(closest_common_time)
gfs_step = list(gfs_valid_times).index(closest_common_time)
print(f"Using common time closest to now ({current_time}):")
print(f"Common time: {closest_common_time}")
print(f"ECMWF step: {ecmwf_step}, GFS step: {gfs_step}")

# Calculate wind speed and convert m/s to knots
ecmwf_wind_speed = 1.94384 * (ecmwf_wind_ds.u ** 2 + ecmwf_wind_ds.v ** 2) ** 0.5
ecmwf_wind_speed.attrs['units'] = 'knots'
ecmwf_wind_speed.attrs['long_name'] = 'Wind Speed'

gfs_wind_speed = 1.94384 * (gfs_ds.u ** 2 + gfs_ds.v ** 2) ** 0.5
gfs_wind_speed.attrs['units'] = 'knots'
gfs_wind_speed.attrs['long_name'] = 'Wind Speed'

# Apply ocean mask to ECMWF wind speed
lsm = ecmwf_land_ds.lsm
ocean_mask = lsm < 0.5
ecmwf_wind_speed = ecmwf_wind_speed.where(ocean_mask)
print("Applied ocean mask to ECMWF wind speed")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ecmwf_wind_speed.isel(step=ecmwf_step).plot(ax=ax1, cmap='viridis')
ax1.set_title(f'ECMWF Wind Speed (Ocean Only)\n{ecmwf_valid_times[ecmwf_step]}')

gfs_wind_speed.isel(step=gfs_step).plot(ax=ax2, cmap='viridis')
ax2.set_title(f'GFS Wind Speed\n{gfs_valid_times[gfs_step]}')
plt.tight_layout()
plt.savefig('wind_speed_comparison.png', dpi=300, bbox_inches='tight')

print("\nComparison plot completed with ocean masking! (Wind speeds in knots)")
```

### Generated Visualizations

![Wind Speed Comparison](images/wind_speed_comparison.png)
*ECMWF vs GFS wind speed comparison with ocean masking*

## GFS Atmospheric Temperature

This example shows how to define a custom model for GFS atmospheric temperature data.

**Script:** [temperature_variation_near_toulouse.py](../examples/temperature_variation_near_toulouse.py)

**Key Features:**
- Defines a custom model configuration for GFS atmospheric data
- Downloads surface temperature data (TMP variable)
- Demonstrates spatial interpolation to a specific location
- Shows temperature unit conversion (Kelvin to Celsius)

**Note:** WindGrib downloads global data and cannot subset by geographic region during download. For single-point analysis like this example, downloading the entire global temperature dataset may be suboptimal compared to specialized point-data APIs.

```python
"""  
WindGrib Usage Example: GFS Atmospheric Temperature Data

This example demonstrates how to define a custom model to download
and analyze GFS atmospheric temperature data.
"""

from matplotlib import pyplot as plt
from windgrib.grib import MODELS, Grib

# Configuration for GFS atmospheric with surface temperature subset
MODELS['gfs_atmos_temperature'] = {
    'product': '.pgrb2.0p25', # start with . to prevent considering goessimpgrb2 product
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {
            'variable': ['TMP'],
            'layer': ['surface']
        }
    },
    'ext': ''
}

print("=== WindGrib Example: GFS Atmospheric Data ===")

print("1. Downloading latest available GFS atmospheric data...")
grib = Grib(model='gfs_atmos_temperature')
grib.download()
print("Download completed")

print("\n2. Analyzing temperature data...")
ds = grib['temperature']
print(f"Available variables: {list(ds.data_vars)}")
print(f"Dimensions: {list(ds.dims)}")

# Convert Kelvin to Celsius and plot temperature variation near Toulouse
temp_celsius = ds.t - 273.15
temp_celsius.attrs['units'] = '°C'
temp_celsius.interp({'latitude':43.599998, 'longitude':1.43333}).plot()
plt.suptitle("Temperature Variation near Toulouse")
plt.savefig("temperature_variation_near_toulouse.png")
print("Example completed")
```

### Generated Visualizations

![Temperature Variation](images/temperature_variation_near_toulouse.png)
*Temperature variation near Toulouse from GFS atmospheric model*