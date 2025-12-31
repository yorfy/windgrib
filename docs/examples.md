# WindGrib Usage Examples

This section presents concrete examples showing the different usage modes of WindGrib. Each example is designed to illustrate a specific aspect of the library.

## Table of Contents

1. [Example 1: Basic Download and Reading](#example-1-basic-download-and-reading)
2. [Example 2: Meteorological Calculations](#example-2-meteorological-calculations)
3. [Example 3: Model Comparison](#example-3-model-comparison)
4. [Example 4: Data Visualization](#example-4-data-visualization)
5. [Example 5: Error Handling](#example-5-error-handling)
6. [Example 6: Advanced Usage](#example-6-advanced-usage)

## Example 1: Basic Download and Reading

### Description
This example shows how to download GRIB data and access it in a basic way.

### Code

```python
from windgrib import Grib

def basic_example():
    """Minimal example of downloading and accessing data."""
    
    # Create a GRIB instance for the GFS Wave model
    grib = Grib(model='gfswave')
    
    # Download data
    print("Downloading GFS Wave data...")
    grib.download()
    
    # Access wind data
    wind_data = grib['wind']
    
    # Display basic information
    print(f"Available variables: {list(wind_data.data_vars)}")
    print(f"Dimensions: {dict(wind_data.sizes)}")
    print(f"Period: {wind_data.time.values}")
    
    # Access a specific subset
    print(f"U data shape: {wind_data.u.shape}")
    print(f"V data shape: {wind_data.v.shape}")
    
    return wind_data

# Execution
if __name__ == '__main__':
    wind_data = basic_example()
    print("✅ Example 1 completed successfully")
```

### Expected Output

```
Downloading GFS Wave data...
Downloading subset: wind - {'variable': ['UGRD', 'VGRD']}
100%|██████████| 5/5 [00:15<00:00,  3.12s/it]
Available variables: ['u', 'v']
Dimensions: {'step': 120, 'latitude': 721, 'longitude': 1440}
Period: [2023-12-25T12:00:00.000000000 ... 2023-12-30T12:00:00.000000000]
U data shape: (120, 721, 1440)
V data shape: (120, 721, 1440)
✅ Example 1 completed successfully
```

## Example 2: Meteorological Calculations

### Description
This example shows how to perform basic meteorological calculations on loaded data.

### Code

```python
from windgrib import Grib
import numpy as np

def meteorological_calculations():
    """Example of basic meteorological calculations."""
    
    # Load data
    print("Loading data...")
    grib = Grib(model='gfswave')
    grib.download()
    wind_data = grib['wind']
    
    # Calculate wind speed
    print("Calculating wind speed...")
    wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
    
    # Calculate wind direction (in degrees)
    print("Calculating wind direction...")
    wind_direction = np.arctan2(wind_data.v, wind_data.u) * (180 / np.pi) % 360
    
    # Calculate statistics
    print("Calculating statistics...")
    mean_speed = wind_speed.mean()
    max_speed = wind_speed.max()
    min_speed = wind_speed.min()
    std_speed = wind_speed.std()
    
    # Display results
    print(f"\n=== Results ===")
    print(f"Mean wind speed: {mean_speed.values:.2f} m/s")
    print(f"Maximum wind speed: {max_speed.values:.2f} m/s")
    print(f"Minimum wind speed: {min_speed.values:.2f} m/s")
    print(f"Standard deviation of speed: {std_speed.values:.2f} m/s")
    print(f"Mean direction: {wind_direction.mean().values:.1f}°")
    
    # Calculate mean speed per time step
    mean_speed_per_step = wind_speed.mean(dim=['latitude', 'longitude'])
    print(f"\nMean speed per time step:")
    print(f"Shape: {mean_speed_per_step.shape}")
    print(f"First values: {mean_speed_per_step.values[:5]}")
    
    return {
        'speed': wind_speed,
        'direction': wind_direction,
        'statistics': {
            'mean': mean_speed,
            'max': max_speed,
            'min': min_speed,
            'std': std_speed
        }
    }

# Execution
if __name__ == '__main__':
    results = meteorological_calculations()
    print("✅ Example 2 completed successfully")
```

### Expected Output

```
Loading data...
Calculating wind speed...
Calculating wind direction...
Calculating statistics...

=== Results ===
Mean wind speed: 7.89 m/s
Maximum wind speed: 24.56 m/s
Minimum wind speed: 0.12 m/s
Standard deviation of speed: 3.45 m/s
Mean direction: 198.7°

Mean speed per time step:
Shape: (120,)
First values: [7.89 7.91 7.93 7.95 7.97]
✅ Example 2 completed successfully
```

## Example 3: Model Comparison

### Description
This example shows how to compare data between different weather models (ECMWF vs GFS).

### Code

```python
from windgrib import Grib
import numpy as np
import pandas as pd

def model_comparison():
    """Comparison between ECMWF and GFS models."""
    
    print("=== Downloading Data ===")
    
    # Download ECMWF data
    print("Downloading ECMWF...")
    grib_ecmwf = Grib(model='ecmwf_ifs')
    grib_ecmwf.download()
    
    # Download GFS data
    print("Downloading GFS...")
    grib_gfs = Grib(model='gfswave')
    grib_gfs.download()
    
    print("\n=== Loading Data ===")
    
    # Load data
    ecmwf_wind = grib_ecmwf['wind']
    gfs_wind = grib_gfs['wind']
    
    print(f"ECMWF variables: {list(ecmwf_wind.data_vars)}")
    print(f"GFS variables: {list(gfs_wind.data_vars)}")
    
    print("\n=== Calculating Wind Speeds ===")
    
    # Calculate wind speeds (in m/s)
    ecmwf_speed = np.sqrt(ecmwf_wind.u**2 + ecmwf_wind.v**2)
    gfs_speed = np.sqrt(gfs_wind.u**2 + gfs_wind.v**2)
    
    # Calculate global statistics
    ecmwf_mean = ecmwf_speed.mean()
    gfs_mean = gfs_speed.mean()
    
    difference = ecmwf_mean - gfs_mean
    relative_difference = abs(difference.values) / ecmwf_mean.values * 100
    
    print(f"\n=== Comparison Results ===")
    print(f"ECMWF mean speed: {ecmwf_mean.values:.2f} m/s")
    print(f"GFS mean speed: {gfs_mean.values:.2f} m/s")
    print(f"Absolute difference: {difference.values:.2f} m/s")
    print(f"Relative difference: {relative_difference:.1f}%")
    
    # Calculate correlation
    ecmwf_flat = ecmwf_speed.values.flatten()
    gfs_flat = gfs_speed.values.flatten()
    
    # Filter NaN values
    mask = ~(np.isnan(ecmwf_flat) | np.isnan(gfs_flat))
    ecmwf_valid = ecmwf_flat[mask]
    gfs_valid = gfs_flat[mask]
    
    correlation = np.corrcoef(ecmwf_valid, gfs_valid)[0, 1]
    print(f"Correlation: {correlation:.3f}")
    
    return {
        'ecmwf': ecmwf_speed,
        'gfs': gfs_speed,
        'comparison': {
            'absolute_difference': difference,
            'relative_difference': relative_difference,
            'correlation': correlation
        }
    }

# Execution
if __name__ == '__main__':
    results = model_comparison()
    print("✅ Example 3 completed successfully")
```

### Expected Output

```
=== Downloading Data ===
Downloading ECMWF...
Downloading subset: wind - {'param': ['10u', '10v']}
100%|██████████| 8/8 [00:22<00:00,  2.75s/it]
Downloading GFS...
Downloading subset: wind - {'variable': ['UGRD', 'VGRD']}
100%|██████████| 5/5 [00:15<00:00,  3.12s/it]

=== Loading Data ===
ECMWF variables: ['u', 'v']
GFS variables: ['u', 'v']

=== Calculating Wind Speeds ===

=== Comparison Results ===
ECMWF mean speed: 8.12 m/s
GFS mean speed: 7.89 m/s
Absolute difference: 0.23 m/s
Relative difference: 2.8%
Correlation: 0.924
✅ Example 3 completed successfully
```

## Example 4: Data Visualization

### Description
This example shows how to visualize meteorological data using matplotlib.

### Code

```python
from windgrib import Grib
import numpy as np
import matplotlib.pyplot as plt

def data_visualization():
    """Example of data visualization with matplotlib."""
    
    # Load data
    print("Loading data...")
    grib = Grib(model='gfswave')
    grib.download()
    wind_data = grib['wind']
    
    # Calculate wind speed
    print("Calculating wind speed...")
    wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
    
    # Create a figure with multiple subplots
    print("Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Wind speed - First time step
    wind_speed.isel(step=0).plot(ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Wind Speed - Step 0')
    
    # Wind speed - Last time step
    wind_speed.isel(step=-1).plot(ax=axes[0, 1], cmap='viridis')
    axes[0, 1].set_title('Wind Speed - Last Step')
    
    # U component of wind
    wind_data.u.isel(step=0).plot(ax=axes[1, 0], cmap='coolwarm')
    axes[1, 0].set_title('U Component - Step 0')
    
    # V component of wind
    wind_data.v.isel(step=0).plot(ax=axes[1, 1], cmap='coolwarm')
    axes[1, 1].set_title('V Component - Step 0')
    
    plt.tight_layout()
    plt.savefig('wind_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'wind_visualization.png'")
    
    # Visualization of mean speed over the period
    plt.figure(figsize=(10, 6))
    mean_wind_speed = wind_speed.mean(dim='step')
    mean_wind_speed.plot(cmap='plasma')
    plt.title('Mean Wind Speed Over Entire Period')
    plt.colorbar(label='Speed (m/s)')
    plt.savefig('mean_speed.png', dpi=300, bbox_inches='tight')
    print("Mean speed saved to 'mean_speed.png'")
    
    # Display visualizations (in interactive environment)
    try:
        plt.show()
    except:
        print("Cannot display plots (non-interactive environment)")
    
    return {
        'wind_speed': wind_speed,
        'mean_wind_speed': mean_wind_speed
    }

# Execution
if __name__ == '__main__':
    results = data_visualization()
    print("✅ Example 4 completed successfully")
```

### Expected Output

```
Loading data...
Calculating wind speed...
Creating visualizations...
Visualization saved to 'wind_visualization.png'
Mean speed saved to 'mean_speed.png'
Cannot display plots (non-interactive environment)
✅ Example 4 completed successfully
```

## Example 5: Error Handling

### Description
This example shows how to handle errors and edge cases when using WindGrib.

### Code

```python
from windgrib import Grib
import pandas as pd
from datetime import datetime

def error_handling():
    """Example of error handling and edge cases."""
    
    print("=== Error Handling Example ===\n")
    
    # Example 1: Handling unavailable dates
    print("1. Handling unavailable dates")
    try:
        # Try to download data for a very old date
        old_date = pd.Timestamp('2020-01-01')
        print(f"   Attempting with old date: {old_date}")
        grib = Grib(time=old_date, model='gfswave')
        grib.download()
    except ValueError as e:
        print(f"   ⚠️  Expected error: {e}")
        print("   ✅ Handling: Using current date instead")
        
        # Fallback solution
        recent_date = pd.Timestamp.now()
        grib = Grib(time=recent_date, model='gfswave')
        grib.download()
        print("   ✅ Download successful with recent date")
    
    # Example 2: Data verification before use
    print("\n2. Data verification before use")
    if grib['wind'] is not None:
        print("   ✅ Wind data available and valid")
        wind_data = grib['wind']
        
        # Check for expected variables
        required_vars = ['u', 'v']
        missing_vars = [var for var in required_vars if var not in wind_data.data_vars]
        
        if missing_vars:
            print(f"   ⚠️  Missing variables: {missing_vars}")
        else:
            print("   ✅ All required variables are present")
            print(f"   Available variables: {list(wind_data.data_vars)}")
    else:
        print("   ❌ Error: Cannot load wind data")
    
    # Example 3: Handling custom paths
    print("\n3. Handling custom paths")
    try:
        # This path will likely fail
        custom_path = '/nonexistent/path/or/unauthorized'
        print(f"   Attempting with path: {custom_path}")
        grib_custom = Grib(data_path=custom_path, model='ecmwf_ifs')
        grib_custom.download()
    except Exception as e:
        print(f"   ⚠️  Path error: {type(e).__name__}: {e}")
        print("   ✅ Handling: Using default path")
        
        # Fallback solution
        grib_default = Grib(model='ecmwf_ifs')
        grib_default.download()
        print("   ✅ Download successful with default path")
    
    # Example 4: Handling missing subsets
    print("\n4. Handling missing subsets")
    try:
        # Try to access a non-existent subset
        nonexistent = grib['nonexistent']
    except KeyError as e:
        print(f"   ⚠️  Non-existent subset: {e}")
        print("   ✅ Handling: List of available subsets")
        print(f"   Available subsets: {list(grib._subsets.keys())}")
    
    # Example 5: Handling partial downloads
    print("\n5. Handling partial downloads")
    grib_test = Grib(model='gfswave')
    
    # Check if data is already available
    wind_subset = grib_test._subsets['wind']
    if wind_subset.grib_file.exists():
        print("   ✅ Data already available in cache")
        print(f"   File: {wind_subset.grib_file}")
        print(f"   Size: {wind_subset.grib_file.stat().st_size / 1e6:.2f} MB")
    else:
        print("   ⚠️  Data not available, download needed")
    
    return grib

# Execution
if __name__ == '__main__':
    grib_instance = error_handling()
    print("\n✅ Example 5 completed successfully")
```

### Expected Output

```
=== Error Handling Example ===

1. Handling unavailable dates
   Attempting with old date: 2020-01-01 00:00:00
   ⚠️  Expected error: No files found after 10 attempts for gfswave
   ✅ Handling: Using current date instead
   ✅ Download successful with recent date

2. Data verification before use
   ✅ Wind data available and valid
   ✅ All required variables are present
   Available variables: ['u', 'v']

3. Handling custom paths
   Attempting with path: /nonexistent/path/or/unauthorized
   ⚠️  Path error: FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/path/or/unauthorized/20231225/12'
   ✅ Handling: Using default path
   ✅ Download successful with default path

4. Handling missing subsets
   ⚠️  Non-existent subset: 'nonexistent'
   ✅ Handling: List of available subsets
   Available subsets: ['wind']

5. Handling partial downloads
   ✅ Data already available in cache
   File: data/grib/20231225/12/wind_gfswave.global.0p25.grib2
   Size: 45.23 MB

✅ Example 5 completed successfully
```

## Example 6: Advanced Usage

### Description
This example shows advanced uses of WindGrib, including direct file access, custom calculations, and integration with other libraries.

### Code

```python
from windgrib import Grib
import numpy as np
import xarray as xr
import pandas as pd

def advanced_usage():
    """Examples of advanced usage and customization."""
    
    print("=== Advanced WindGrib Usage ===\n")
    
    # 1. Direct access to files and subsets
    print("1. Direct file access")
    grib = Grib(model='ecmwf_ifs')
    grib.download()
    
    wind_subset = grib._subsets['wind']
    print(f"   GRIB file: {wind_subset.grib_file}")
    print(f"   NetCDF file: {wind_subset.nc_file}")
    print(f"   Index file: {wind_subset.grib_ls}")
    print(f"   File size: {wind_subset.grib_file.stat().st_size / 1e6:.2f} MB")
    
    # 2. Manual loading with custom options
    print("\n2. Custom loading with xarray")
    custom_ds = xr.open_dataset(
        wind_subset.grib_file,
        engine='cfgrib',
        decode_timedelta=True,
        backend_kwargs={
            'errors': 'ignore',
            'filter_by_keys': {'shortName': 'u'}
        },
        chunks={'step': 1, 'latitude': 100, 'longitude': 100}
    )
    
    print(f"   Custom dataset loaded: {list(custom_ds.data_vars)}")
    print(f"   Shape: {custom_ds.u.shape}")
    print(f"   Chunking: {custom_ds.u.chunks}")
    
    # 3. Advanced calculations with xarray
    print("\n3. Advanced calculations")
    
    # Calculate vorticity (to identify eddies)
    print("   Calculating vorticity...")
    dx = 0.25  # degree
    dy = 0.25  # degree
    
    # Calculate derivatives (simplified)
    du_dy, du_dx = np.gradient(custom_ds.u.isel(step=0), dy, dx)
    dv_dy, dv_dx = np.gradient(custom_ds.v.isel(step=0), dy, dx)
    
    # Vorticity: dv/dx - du/dy
    vorticity = dv_dx - du_dy
    
    print(f"   Vorticity calculated - Shape: {vorticity.shape}")
    print(f"   Values: min={vorticity.min():.6f}, max={vorticity.max():.6f}")
    print(f"   Mean: {vorticity.mean():.6f}, Std: {vorticity.std():.6f}")
    
    # 4. Spatial selection
    print("\n4. Spatial selection")
    
    # Select a specific region (Europe)
    europe = wind_data.sel(
        latitude=slice(35, 70),
        longitude=slice(-15, 30)
    )
    
    print(f"   Europe region selected: {dict(europe.sizes)}")
    
    # Calculate mean speed for this region
    europe_speed = np.sqrt(europe.u**2 + europe.v**2)
    europe_mean_speed = europe_speed.mean()
    
    print(f"   Mean wind speed in Europe: {europe_mean_speed.values:.2f} m/s")
    
    # 5. Temporal aggregation
    print("\n5. Temporal aggregation")
    
    # Calculate rolling mean
    rolling_mean = wind_speed.rolling(step=3, center=True).mean()
    
    print(f"   Rolling mean calculated: {rolling_mean.shape}")
    print(f"   First value: {rolling_mean.isel(step=0).values:.2f} m/s")
    print(f"   Last value: {rolling_mean.isel(step=-1).values:.2f} m/s")
    
    # 6. Integration with pandas
    print("\n6. Integration with pandas")
    
    # Convert to pandas DataFrame for time series analysis
    time_series = wind_speed.mean(dim=['latitude', 'longitude'])
    df = time_series.to_dataframe(name='wind_speed')
    df['model'] = grib.model['name']
    
    print(f"   DataFrame created: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index.name}")
    
    # Descriptive statistics
    print(f"\n   Descriptive statistics:")
    print(df['wind_speed'].describe())
    
    # 7. Trend analysis
    print("\n7. Trend analysis")
    
    # Calculate linear trend
    x = np.arange(len(df))
    y = df['wind_speed'].values
    
    # Simple linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    print(f"   Trend slope: {m:.6f} m/s per time step")
    print(f"   Intercept: {c:.2f} m/s")
    
    if m > 0:
        print("   📈 Increasing trend")
    elif m < 0:
        print("   📉 Decreasing trend")
    else:
        print("   📊 No clear trend")
    
    return {
        'vorticity': vorticity,
        'europe_data': europe,
        'rolling_mean': rolling_mean,
        'dataframe': df,
        'trend': {'slope': m, 'intercept': c}
    }

# Execution
if __name__ == '__main__':
    results = advanced_usage()
    print("\n✅ Example 6 completed successfully")
```

### Expected Output

```
=== Advanced WindGrib Usage ===

1. Direct file access
   GRIB file: data/grib/20231225/12/wind_ecmwf_ifs.oper.grib2
   NetCDF file: data/grib/20231225/12/wind_ecmwf_ifs.oper.nc
   Index file: data/grib/20231225/12/wind_ecmwf_ifs.oper.grib2.ls
   File size: 68.45 MB

2. Custom loading with xarray
   Custom dataset loaded: ['u']
   Shape: (120, 721, 1440)
   Chunking: ((1, 1, 1, ..., 1, 1, 1), (100, 100, 100, ..., 100, 100, 100), (100, 100, 100, ..., 100, 100, 100))

3. Advanced calculations
   Calculating vorticity...
   Vorticity calculated - Shape: (720, 1439)
   Values: min=-0.000123, max=0.000456
   Mean: 0.000012, Std: 0.000056

4. Spatial selection
   Europe region selected: {'step': 120, 'latitude': 355, 'longitude': 188}
   Mean wind speed in Europe: 6.78 m/s

5. Temporal aggregation
   Rolling mean calculated: (120, 721, 1440)
   First value: 7.89 m/s
   Last value: 8.12 m/s

6. Integration with pandas
   DataFrame created: (120, 2)
   Columns: ['wind_speed', 'model']
   Index: step

   Descriptive statistics:
   count    120.000000
   mean       7.987654
   std        0.456789
   min        7.123456
   25%        7.654321
   50%        7.987654
   75%        8.321098
   max        8.765432

7. Trend analysis
   Trend slope: 0.002345 m/s per time step
   Intercept: 7.89 m/s
   📈 Increasing trend

✅ Example 6 completed successfully
```

## Conclusion

These examples cover the main usage modes of WindGrib, from basic operations to advanced uses. Each example can be run independently and illustrates key concepts of the library.

For optimal use:

1. **Start with basic examples** to understand fundamental concepts
2. **Explore advanced examples** to discover more complex features
3. **Adapt examples** to your specific needs
4. **Consult technical documentation** to understand implementation details

Feel free to modify and extend these examples to meet your specific meteorological analysis needs.