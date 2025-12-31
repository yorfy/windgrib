# WindGrib Usage Examples

This section presents concrete examples showing different usage modes of WindGrib. Each example is designed to illustrate a specific aspect of the library.

## Table of Contents

1. [Example 1: Basic Download and Reading](#example-1-basic-download-and-reading)
2. [Example 2: Meteorological Calculations](#example-2-meteorological-calculations)
3. [Example 3: Model Comparison](#example-3-model-comparison)
4. [Example 4: Data Visualization](#example-4-data-visualization)
5. [Example 5: Error Handling](#example-5-error-handling)
6. [Example 6: Advanced Usage](#example-6-advanced-usage)
7. [Example 7: GFS Atmospheric Temperature](#example-7-gfs-atmospheric-temperature)

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
    
    # Download the data
    print("Downloading GFS Wave data...")
    grib.download()
    
    # Access wind data
    wind_data = grib['wind']
    
    # Display basic information
    print(f"Available variables: {list(wind_data.data_vars)}")
    print(f"Dimensions: {dict(wind_data.sizes)}")
    print(f"Time period: {wind_data.time.values}")
    
    # Access specific subset
    print(f"U data shape: {wind_data.u.shape}")
    print(f"V data shape: {wind_data.v.shape}")
    
    return wind_data

# Execution
if __name__ == '__main__':
    wind_data = basic_example()
    print("✅ Example 1 completed successfully")
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
    print(f"Average wind speed: {mean_speed.values:.2f} m/s")
    print(f"Maximum wind speed: {max_speed.values:.2f} m/s")
    print(f"Minimum wind speed: {min_speed.values:.2f} m/s")
    print(f"Speed standard deviation: {std_speed.values:.2f} m/s")
    print(f"Average direction: {wind_direction.mean().values:.1f}°")
    
    return {
        'speed': wind_speed,
        'direction': wind_direction,
        'statistics': {'mean': mean_speed, 'max': max_speed, 'min': min_speed, 'std': std_speed}
    }

# Execution
if __name__ == '__main__':
    results = meteorological_calculations()
    print("✅ Example 2 completed successfully")
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
    
    print("=== Data Download ===")
    
    # Download ECMWF data
    print("Downloading ECMWF...")
    grib_ecmwf = Grib(model='ecmwf_ifs')
    grib_ecmwf.download()
    
    # Download GFS data
    print("Downloading GFS...")
    grib_gfs = Grib(model='gfswave')
    grib_gfs.download()
    
    print("\n=== Data Loading ===")
    
    # Load data
    ecmwf_wind = grib_ecmwf['wind']
    gfs_wind = grib_gfs['wind']
    
    print(f"ECMWF variables: {list(ecmwf_wind.data_vars)}")
    print(f"GFS variables: {list(gfs_wind.data_vars)}")
    
    print("\n=== Wind Speed Calculation ===")
    
    # Calculate wind speeds (in m/s)
    ecmwf_speed = np.sqrt(ecmwf_wind.u**2 + ecmwf_wind.v**2)
    gfs_speed = np.sqrt(gfs_wind.u**2 + gfs_wind.v**2)
    
    # Calculate global statistics
    ecmwf_mean = ecmwf_speed.mean()
    gfs_mean = gfs_speed.mean()
    
    difference = ecmwf_mean - gfs_mean
    relative_difference = abs(difference.values) / ecmwf_mean.values * 100
    
    print(f"\n=== Comparison Results ===")
    print(f"ECMWF average speed: {ecmwf_mean.values:.2f} m/s")
    print(f"GFS average speed: {gfs_mean.values:.2f} m/s")
    print(f"Absolute difference: {difference.values:.2f} m/s")
    print(f"Relative difference: {relative_difference:.1f}%")
    
    return {
        'ecmwf': ecmwf_speed,
        'gfs': gfs_speed,
        'comparison': {
            'absolute_difference': difference,
            'relative_difference': relative_difference
        }
    }

# Execution
if __name__ == '__main__':
    results = model_comparison()
    print("✅ Example 3 completed successfully")
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
    
    # Create figure with multiple subplots
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
    
    # Visualization of average wind speed over period
    plt.figure(figsize=(10, 6))
    mean_wind_speed = wind_speed.mean(dim='step')
    mean_wind_speed.plot(cmap='plasma')
    plt.title('Average Wind Speed Over Entire Period')
    plt.colorbar(label='Speed (m/s)')
    plt.savefig('average_speed.png', dpi=300, bbox_inches='tight')
    print("Average speed saved to 'average_speed.png'")
    
    # Display visualizations (in interactive environment)
    try:
        plt.show()
    except:
        print("Cannot display graphics (non-interactive environment)")
    
    return {
        'wind_speed': wind_speed,
        'mean_wind_speed': mean_wind_speed
    }

# Execution
if __name__ == '__main__':
    results = data_visualization()
    print("✅ Example 4 completed successfully")
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
    
    # Example 2: Data validation before use
    print("\n2. Data validation before use")
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
        print("   ❌ Error: Unable to load wind data")
    
    # Example 3: Custom path handling
    print("\n3. Custom path handling")
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
    
    return grib

# Execution
if __name__ == '__main__':
    grib_instance = error_handling()
    print("\n✅ Example 5 completed successfully")
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
    
    # 1. Direct file and subset access
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
    
    # Calculate vorticity (to identify vortices)
    print("   Calculating vorticity...")
    dx = 0.25  # degrees
    dy = 0.25  # degrees
    
    # Calculate derivatives (simplified)
    du_dy, du_dx = np.gradient(custom_ds.u.isel(step=0), dy, dx)
    dv_dy, dv_dx = np.gradient(custom_ds.v.isel(step=0), dy, dx)
    
    # Vorticity: dv/dx - du/dy
    vorticity = dv_dx - du_dy
    
    print(f"   Vorticity calculated - Shape: {vorticity.shape}")
    print(f"   Values: min={vorticity.min():.6f}, max={vorticity.max():.6f}")
    print(f"   Mean: {vorticity.mean():.6f}, Std: {vorticity.std():.6f}")
    
    return {
        'vorticity': vorticity,
        'custom_dataset': custom_ds
    }

# Execution
if __name__ == '__main__':
    results = advanced_usage()
    print("\n✅ Example 6 completed successfully")
```

## Conclusion

These examples cover the main usage modes of WindGrib, from basic operations to advanced uses. Each example can be run independently and illustrates key concepts of the library.

For optimal use:

1. **Start with basic examples** to understand fundamental concepts
2. **Explore advanced examples** to discover more complex features
3. **Adapt examples** to your specific needs
4. **Consult technical documentation** to understand implementation details

Feel free to modify and extend these examples to meet your specific meteorological analysis needs.

## Example 7: GFS Atmospheric Temperature

### Description
This example shows how to define a custom model for GFS atmospheric temperature data and perform temperature analysis.

### Code

```python
from windgrib.grib import MODELS, Grib
import numpy as np

def gfs_temperature_example():
    """Example of using GFS atmospheric temperature data."""
    
    print("=== GFS Atmospheric Temperature Example ===\n")
    
    # Define the GFS atmospheric model
    print("1. Defining GFS atmospheric model...")
    MODELS['gfs_atmos'] = {
        'product': 'global.0p25',
        'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
        'key': 'gfs.{date}/{h:02d}/atmos/',
        'subsets': {
            'temperature': {
                'variable': ['TMP']  # Temperature in Kelvin
            }
        }
    }
    
    # Initialize and download
    print("2. Downloading GFS atmospheric data...")
    grib = Grib(
        model='gfs_atmos',
        time='2023-12-25 12:00',  # Example date
        data_path='./data/gfs_atmos'
    )
    
    try:
        grib.download()
        print("   ✅ Download completed")
    except Exception as e:
        print(f"   ⚠️  Download error: {e}")
        print("   Using fallback approach...")
        # Fallback to current time
        grib = Grib(model='gfs_atmos')
        grib.download()
    
    # Access temperature data
    print("\n3. Analyzing temperature data...")
    temp_data = grib['temperature']
    
    print(f"   Available variables: {list(temp_data.data_vars)}")
    print(f"   Data dimensions: {dict(temp_data.sizes)}")
    
    # Convert from Kelvin to Celsius
    temp_kelvin = temp_data.TMP
    temp_celsius = temp_kelvin - 273.15
    
    # Calculate statistics
    print("\n4. Temperature statistics:")
    print(f"   Mean temperature: {temp_celsius.mean().values:.2f}°C")
    print(f"   Min temperature: {temp_celsius.min().values:.2f}°C")
    print(f"   Max temperature: {temp_celsius.max().values:.2f}°C")
    print(f"   Standard deviation: {temp_celsius.std().values:.2f}°C")
    
    # Regional analysis (example: Europe)
    print("\n5. Regional analysis (Europe):")
    try:
        europe_temp = temp_celsius.sel(
            latitude=slice(70, 35),
            longitude=slice(-10, 40)
        )
        print(f"   Europe mean temperature: {europe_temp.mean().values:.2f}°C")
        print(f"   Europe temperature range: {europe_temp.min().values:.2f}°C to {europe_temp.max().values:.2f}°C")
    except Exception as e:
        print(f"   Regional analysis not available: {e}")
    
    # Temporal analysis (if multiple time steps)
    print("\n6. Temporal analysis:")
    if 'step' in temp_celsius.dims and len(temp_celsius.step) > 1:
        # Calculate global mean for each time step
        global_mean_temp = temp_celsius.mean(dim=['latitude', 'longitude'])
        
        print("   Temperature evolution:")
        for i, step in enumerate(global_mean_temp.step.values[:5]):  # First 5 steps
            temp_val = global_mean_temp.isel(step=i).values
            print(f"     Step {i}: {temp_val:.2f}°C")
        
        # Calculate trend
        temps = global_mean_temp.values
        if len(temps) > 1:
            trend = np.polyfit(range(len(temps)), temps, 1)[0]
            if abs(trend) > 0.01:
                trend_direction = "warming" if trend > 0 else "cooling"
                print(f"   Trend: {trend_direction} ({trend:.3f}°C/step)")
            else:
                print("   Trend: stable")
    else:
        print("   Single time step - no temporal analysis")
    
    # Convert to NetCDF
    print("\n7. Converting to NetCDF...")
    try:
        grib.to_nc()
        print("   ✅ NetCDF conversion completed")
    except Exception as e:
        print(f"   ⚠️  NetCDF conversion error: {e}")
    
    return {
        'temperature_celsius': temp_celsius,
        'statistics': {
            'mean': temp_celsius.mean().values,
            'min': temp_celsius.min().values,
            'max': temp_celsius.max().values,
            'std': temp_celsius.std().values
        }
    }

def temperature_comparison():
    """Compare temperature data from different sources or times."""
    
    print("\n=== Temperature Comparison Example ===\n")
    
    # Load temperature data for two different times
    print("Loading temperature data for comparison...")
    
    try:
        # Current data
        grib_current = Grib(model='gfs_atmos')
        grib_current.download()
        temp_current = grib_current['temperature'].TMP - 273.15
        
        # Data from 6 hours earlier (if available)
        import pandas as pd
        earlier_time = pd.Timestamp.now() - pd.Timedelta(hours=6)
        grib_earlier = Grib(model='gfs_atmos', time=earlier_time)
        grib_earlier.download()
        temp_earlier = grib_earlier['temperature'].TMP - 273.15
        
        # Calculate difference
        temp_diff = temp_current - temp_earlier
        
        print("Temperature change analysis:")
        print(f"  Mean change: {temp_diff.mean().values:.2f}°C")
        print(f"  Max warming: {temp_diff.max().values:.2f}°C")
        print(f"  Max cooling: {temp_diff.min().values:.2f}°C")
        
        # Identify regions with significant changes
        significant_warming = temp_diff > 2.0  # More than 2°C warming
        significant_cooling = temp_diff < -2.0  # More than 2°C cooling
        
        warming_percent = (significant_warming.sum() / temp_diff.size * 100).values
        cooling_percent = (significant_cooling.sum() / temp_diff.size * 100).values
        
        print(f"  Significant warming areas: {warming_percent:.1f}%")
        print(f"  Significant cooling areas: {cooling_percent:.1f}%")
        
    except Exception as e:
        print(f"Comparison not available: {e}")

# Execution
if __name__ == '__main__':
    # Run main temperature example
    results = gfs_temperature_example()
    
    # Run comparison example
    temperature_comparison()
    
    print("\n✅ Example 7 completed successfully")
```

### Key Features Demonstrated

1. **Custom Model Definition**: Shows how to define a new model for GFS atmospheric data
2. **Temperature Analysis**: Conversion from Kelvin to Celsius and statistical calculations
3. **Regional Analysis**: Extracting data for specific geographic regions
4. **Temporal Analysis**: Analyzing temperature trends over time
5. **Error Handling**: Robust handling of download and processing errors
6. **Data Comparison**: Comparing temperature data across different times

### Usage Notes

- The GFS atmospheric data path follows the pattern: `gfs.{date}/{hour}/atmos/`
- Temperature data is provided in Kelvin and needs conversion to Celsius
- The model can be extended to include other atmospheric variables like humidity, pressure, etc.
- Regional analysis requires understanding of the coordinate system used in the data