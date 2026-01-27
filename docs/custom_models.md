# Creating Custom Models with WindGrib

This guide explains how to extend WindGrib to support new data types by creating custom models.

## Table of Contents

1. [Model Structure](#model-structure)
2. [Basic Example](#basic-example)
3. [Advanced Configuration](#advanced-configuration)

## Model Structure

A WindGrib model is defined by a dictionary with the following structure:

```python
MODELS['my_model'] = {
    'product': 'product_id',            # Product identifier
    'url': 'https://source.com/',       # Base URL
    'key': '{date}/{h:02d}/path/',     # Relative path (with formatting)
    'idx': '.idx',                     # Index file extension (optional)
    'subsets': {                       # Data subsets
        'subset_name': ['VAR1', 'VAR2']  # Variables to download
    }
}
```

## Basic Example

Here's how to add a model for downloading GFS atmospheric temperature data:

```python
from windgrib.grib import MODELS, Grib

# Add a model for GFS temperature data
MODELS['gfs_temp'] = {
    'product': '.pgrb2.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': ['TMP']  # Temperature variable
    },
    'ext': ''
}

if __name__ == '__main__':
    # Use the new model
    gb = Grib(model='gfs_temp')
    gb.download()

    # Access the data
    temp_data = gb['temperature'].ds
    print(f"Mean temperature: {temp_data.t.mean().values:.2f} K")
```

## Advanced Configuration

### Multiple Variables and Filters

You can define subsets with multiple variables and apply filters:

```python
MODELS['gfs_advanced'] = {
    'product': '.pgrb2.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'surface_temp': (['TMP'], {'layer': ['surface']}),  # Surface only
        'wind_850': (['UGRD', 'VGRD'], {'level': ['850']})  # 850 hPa level
    },
    'ext': ''
}
```

### Subset Configuration Options

Subsets can be defined in multiple ways:

```python
# Simple: list of variables
'wind': ['UGRD', 'VGRD']

# With filters: tuple of (variables, filters)
'surface_temp': (['TMP'], {'layer': ['surface']})

# With step: tuple of (variables, step)
'land_mask': (['lsm'], 0)  # Only step 0

# With both: tuple of (variables, step, filters)
'pressure': (['PRMSL'], 0, {'layer': ['surface']})
```

### Subset with Temperature Example

This example from [temperature_variation_near_toulouse.py](../examples/temperature_variation_near_toulouse.py) shows a complete custom model implementation:

**Note:** WindGrib downloads global data and cannot subset by geographic region during download. For single-point analysis like this example, downloading the entire global temperature dataset may be suboptimal compared to specialized point-data APIs.

```python
from matplotlib import pyplot as plt
from windgrib.grib import MODELS, Grib

# Configuration for GFS atmospheric with surface temperature subset
MODELS['gfs_atmos_temperature'] = {
    'product': '.pgrb2.0p25',  # start with . to prevent considering goessimpgrb2 product
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': (['TMP'], {'layer': ['surface']})
    },
    'ext': ''
}

if __name__ == '__main__':
    print("=== WindGrib Example: GFS Atmospheric Data ===\n")

    print("1. Downloading latest available GFS atmospheric data...")
    gb = Grib(model='gfs_atmos_temperature')
    gb.download()
    print("Download completed")

    print("\n2. Analyzing temperature data...")
    ds = gb['temperature'].ds
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {list(ds.dims)}")

    # Convert Kelvin to Celsius and plot temperature variation near Toulouse
    temp_celsius = ds.t - 273.15
    temp_celsius.attrs['units'] = '°C'
    temp_celsius.interp({'latitude': 43.599998, 'longitude': 1.43333}).plot()
    plt.suptitle("Temperature Variation near Toulouse")
    plt.savefig("../docs/images/temperature_variation_near_toulouse.png")
    print("Example completed")
```

#### Generated Visualization

![Temperature Variation](images/temperature_variation_near_toulouse.png)
*Temperature variation near Toulouse from GFS atmospheric model*

#### Key Points

1. **Product Identifier**: Use `.pgrb2.0p25` to avoid matching unwanted products
2. **Subset Filters**: Apply `{'layer': ['surface']}` to get only surface data
3. **Empty Extension**: Set `'ext': ''` when files don't have a standard extension
4. **Data Access**: Variables are accessible via their short names (e.g., `ds.t` for temperature)

### Subset with multiple levels indexing

When working with atmospheric data at multiple pressure levels, you can use filter keys to select specific levels and xarray operations to restructure the dataset.


```python
from windgrib.grib import MODELS, Grib

# Define model with temperature variable
MODELS['ecmwf_t'] = {
    'product': 'oper',
    'url': 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/',
    'key': '{date}/{h:02d}z/ifs/0p25/oper/',
    'idx': '.index',
    'subsets': {'temperature': 't'}
}

if __name__ == '__main__':
    # Download temperature data at specific pressure levels
    gb = Grib(model='ecmwf_t')
    subset = gb['temperature'].sel(levelist=['50', '100', '200'])
    subset.download()

    # Restructure dataset to have levels as a separate dimension
    ds = subset.ds
    ds = ds.assign_coords(step_original=ds.step)
    ds = ds.set_index(step=['step_original', 'isobaricInhPa']).unstack('step')
    ds = ds.rename({'step_original': 'step'})

    # Result: dataset with dimensions (step, isobaricInhPa, latitude, longitude)
    print(ds)
```

#### Key Techniques

1. **Filter by Level**: Use `sel(levelist=['50', '100', '200'])` to download specific pressure levels
2. **Preserve Original Step**: `assign_coords(step_original=ds.step)` saves the time dimension
3. **Create Multi-Index**: `set_index(step=['step_original', 'isobaricInhPa'])` combines step and level
4. **Unstack Dimensions**: `unstack('step')` separates the multi-index into distinct dimensions
5. **Rename Coordinates**: `rename({'step_original': 'step'})` restores the original dimension name
