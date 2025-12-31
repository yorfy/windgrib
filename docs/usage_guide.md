# WindGrib Usage Guide

This guide explains the different usage modes of the WindGrib library for working with GRIB (GRIdded Binary) data.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Supported Data Models](#supported-data-models)
4. [Main Usage Modes](#main-usage-modes)
   - [GRIB Data Download](#grib-data-download)
   - [Data Reading and Loading](#data-reading-and-loading)
   - [NetCDF Conversion](#netcdf-conversion)
   - [Data Subset Access](#data-subset-access)
   - [Data Processing and Analysis](#data-processing-and-analysis)
5. [Advanced Examples](#advanced-examples)
6. [Error Handling and Best Practices](#error-handling-and-best-practices)

## Introduction

WindGrib is a Python library for downloading, reading, and processing meteorological data in GRIB format. It supports multiple weather models and provides a simple interface for working with these complex data.

## Installation

```bash
pip install windgrib
```

## Supported Data Models

WindGrib currently supports the following models:

### 1. GFS Wave (Global Forecast System)
- **Identifier**: `gfswave`
- **Product**: `global.0p25`
- **Available Variables**: UGRD (U wind component), VGRD (V wind component)
- **Resolution**: 0.25 degrees
- **Source**: NOAA

### 2. ECMWF IFS (European Centre for Medium-Range Weather Forecasts)
- **Identifier**: `ecmwf_ifs`
- **Product**: `oper`
- **Available Variables**:
  - Wind: 10u (U component), 10v (V component)
  - Land: lsm (land/sea mask)
- **Resolution**: 0.25 degrees
- **Source**: ECMWF

## Main Usage Modes

### GRIB Data Download

The most common mode is downloading GRIB data from remote sources.

#### Basic Example

```python
from windgrib import Grib

# Create a GRIB instance for the GFS Wave model
grib = Grib(model='gfswave')

# Download the data
grib.download()
```

#### Advanced Options

```python
from windgrib import Grib
from datetime import datetime

# Specify a specific date and model
grib = Grib(
    time=datetime(2023, 12, 25, 12),  # Specific date and time
    model='ecmwf_ifs',                # ECMWF model
    data_path='./my_data',            # Custom path for data
    max_retry=5                       # Maximum number of attempts
)

# Download without using cache
grib.download(use_cache=False)
```

### Data Reading and Loading

Once data is downloaded, you can load it into xarray objects for analysis.

```python
from windgrib import Grib

# Create instance and download
grib = Grib(model='gfswave')
grib.download()

# Access data from a specific subset
wind_data = grib['wind']

# Display dataset information
print(f"Available variables: {list(wind_data.data_vars)}")
print(f"Dimensions: {dict(wind_data.sizes)}")
```

### NetCDF Conversion

For more efficient use, you can convert GRIB files to NetCDF format.

```python
from windgrib import Grib

# Create instance and download
grib = Grib(model='ecmwf_ifs')
grib.download()

# Convert all subsets to NetCDF
grib.to_nc()

# Convert a specific subset
wind_subset = grib._subsets['wind']
wind_subset.to_nc()
```

### Data Subset Access

WindGrib organizes data into logical subsets for easy access.

```python
from windgrib import Grib

# Create ECMWF instance
grib = Grib(model='ecmwf_ifs')
grib.download()

# Access wind subset
wind_data = grib['wind']

# Access land data subset
land_data = grib['land']

# Work with the data
print(f"Wind variables: {list(wind_data.data_vars)}")
print(f"Land variables: {list(land_data.data_vars)}")
```

### Data Processing and Analysis

WindGrib integrates well with xarray and numpy for data processing.

```python
import numpy as np
from windgrib import Grib

# Load data
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Calculate wind speed from components
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)

# Calculate average wind speed over time
mean_wind_speed = wind_speed.mean(dim='step')

# Find maximum wind speed
max_wind_speed = wind_speed.max()

print(f"Average wind speed: {mean_wind_speed.values}")
print(f"Maximum wind speed: {max_wind_speed.values}")
```

## Advanced Examples

### Model Comparison

```python
from windgrib import Grib
import numpy as np

# Download ECMWF and GFS data
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_ecmwf.download()
grib_ecmwf.to_nc()

grib_gfs = Grib(model='gfswave')
grib_gfs.download()
grib_gfs.to_nc()

# Load data
ecmwf_wind = grib_ecmwf['wind']
gfs_wind = grib_gfs['wind']

# Calculate wind speeds (in knots)
ecmwf_speed = (ecmwf_wind.u**2 + ecmwf_wind.v**2)**0.5 * 1.94384
gfs_speed = (gfs_wind.u**2 + gfs_wind.v**2)**0.5 * 1.94384

# Compare forecasts
difference = ecmwf_speed - gfs_speed
print(f"Average difference: {difference.mean().values}")
```

### Visualization with matplotlib

```python
from windgrib import Grib
import matplotlib.pyplot as plt

# Load data
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Calculate wind speed
wind_speed = (wind_data.u**2 + wind_data.v**2)**0.5

# Visualize wind speed for first time step
wind_speed.isel(step=0).plot(cmap='viridis')
plt.title('Wind Speed - First Time Step')
plt.show()
```

### Custom Data Paths

```python
from windgrib import Grib
from pathlib import Path

# Use custom path for data
custom_path = Path('./my_weather_data')
grib = Grib(data_path=custom_path, model='ecmwf_ifs')

# Download data to custom path
grib.download()

# Files will be stored in ./my_weather_data/YYYYMMDD/HH/
```

## Error Handling and Best Practices

### Common Error Handling

```python
from windgrib import Grib
import pandas as pd

try:
    # Try to download data for a very old date
    old_date = pd.Timestamp('2020-01-01')
    grib = Grib(time=old_date, model='gfswave')
    grib.download()
except ValueError as e:
    print(f"Error: {e}")
    # Handle error by using a more recent date
    recent_date = pd.Timestamp.now()
    grib = Grib(time=recent_date, model='gfswave')
    grib.download()
```

### Best Practices

1. **Always use cache**: GRIB data download can be slow. Use `use_cache=True` (default) to avoid downloading already available data.

2. **Memory management**: GRIB data can be large. Use chunking methods to manage memory.

3. **Data cleanup**: After use, you can delete temporary files to free disk space.

4. **Data validation**: Always check that data was correctly downloaded before using it.

```python
from windgrib import Grib

# Example of data validation
grib = Grib(model='ecmwf_ifs')
grib.download()

# Check that data is available
if grib['wind'] is not None:
    print("Wind data available")
    # Process data
else:
    print("Error: wind data not available")
```

## Conclusion

WindGrib offers a powerful and flexible interface for working with GRIB data. Whether you need to download, read, convert, or analyze meteorological data, this library provides the necessary tools to work efficiently with these complex data formats.

For more information, consult the [complete documentation](link_to_full_docs) and [examples](link_to_examples).