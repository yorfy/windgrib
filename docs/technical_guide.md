# WindGrib Technical Guide

This technical guide explains the implementation details and advanced usage modes of the WindGrib library.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Main Classes](#main-classes)
3. [Download Mechanisms](#download-mechanisms)
4. [Subset Management](#subset-management)
5. [Format Conversion](#format-conversion)
6. [xarray Integration](#xarray-integration)
7. [Error Handling](#error-handling)
8. [Performance Optimizations](#performance-optimizations)
9. [Extensibility](#extensibility)

## System Architecture

WindGrib follows a modular architecture with the following main components:

```
┌───────────────────────────────────────────────────┐
│                 WindGrib Library                  │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────┐    ┌─────────────────┐           │
│  │   Grib      │    │  GribSubset     │           │
│  │ (Main       │    │ (Subset         │           │
│  │  class)     │    │  class)         │           │
│  └─────────────┘    └─────────────────┘           │
│          │                     │                  │
│          ▼                     ▼                  │
│  ┌─────────────┐    ┌─────────────────┐           │
│  │  Download   │    │   Data Loading  │           │
│  │  Manager    │    │   & Processing  │           │
│  └─────────────┘    └─────────────────┘           │
│          │                     │                  │
│          ▼                     ▼                  │
│  ┌─────────────┐    ┌─────────────────┐           │
│  │  S3/HTTP    │    │   xarray/cfgrib │           │
│  │  Interface  │    │   Integration   │           │
│  └─────────────┘    └─────────────────┘           │
│                                                   │
└───────────────────────────────────────────────────┘
```

## Main Classes

### Grib Class

The main class that manages global configuration and high-level operations.

**Main Attributes:**
- `model`: Weather model configuration
- `date`: Data date
- `h`: Data hour
- `data_path`: Data storage path
- `_subsets`: Dictionary of subsets

**Key Methods:**
- `__init__()`: Initialization with configuration
- `download()`: Download all subsets
- `to_nc()`: Convert all subsets to NetCDF
- `__getitem__()`: Access subset data

### GribSubset Class

Manages operations specific to a data subset.

**Main Attributes:**
- `name`: Subset name
- `config`: Specific configuration
- `grib`: Reference to parent Grib instance
- `_ds`: Cached xarray dataset

**Key Methods:**
- `download()`: Download files for this subset
- `load_dataset()`: Load dataset
- `to_nc()`: Convert to NetCDF
- `messages_df()`: Analyze index files

## Download Mechanisms

### Download Process

1. **File Detection**: Use `s3fs` to list available files
2. **Filtering**: Select relevant files for the subset
3. **Batch Download**: Use `ThreadPoolExecutor` for parallelism
4. **Cache Management**: Check existing files before download
5. **Error Recovery**: Retry mechanism with `max_retry`

### Download Flow Example

```python
# 1. Initialization
grib = Grib(model='ecmwf_ifs')

# 2. Detect available files
grib.find_forecast_time()

# 3. Download subsets
for subset in grib._subsets.values():
    subset.download()
    
# 4. For each subset:
#    - Check cache
#    - Download missing files
#    - Update index files
```

### HTTP Request Management

The system uses HTTP requests with byte ranges to download only necessary parts of GRIB files:

```python
headers = {'Range': f"bytes={start_byte}-{end_byte}"}
r = get(url, headers=headers, timeout=30)
```

## Subset Management

### Subset Structure

Each model defines its subsets in the `MODELS` configuration:

```python
MODELS = {
    'gfswave': {
        'subsets': {
            'wind': {'variable': ['UGRD', 'VGRD']}
        }
    },
    'ecmwf_ifs': {
        'subsets': {
            'wind': {'param': ['10u', '10v']},
            'land': {'param': ['lsm'], 'step': 0}
        }
    }
}
```

### Subset Access

```python
# Direct access via bracket notation
grib = Grib(model='ecmwf_ifs')
wind_data = grib['wind']  # Returns xarray Dataset

# Access via subset object
wind_subset = grib._subsets['wind']
dataset = wind_subset.ds  # Property that loads dataset
```

## Format Conversion

### GRIB → NetCDF Conversion Process

1. **Dataset Loading**: Use `xr.open_dataset()` with cfgrib
2. **Memory Optimization**: Chunked loading
3. **Data Encoding**: Apply optimized encoding
4. **Saving**: Use `to_netcdf()` with compression

### Conversion Example

```python
# Convert specific subset
subset = grib._subsets['wind']

# Load dataset with chunking
ds = subset.load_grib_file()

# Apply encoding
encoding = {
    var: {
        'dtype': 'int16', 
        'scale_factor': 0.01,
        '_FillValue': np.iinfo('int16').max, 
        'zlib': False
    }
    for var in ds.data_vars
}

# Save as NetCDF
ds.to_netcdf(subset.nc_file, encoding=encoding)
```

### NetCDF Format Advantages

- **Faster Reading**: Direct data access without GRIB parsing
- **Compatibility**: Widely supported standard format
- **Reduced Size**: Efficient data compression
- **Preserved Metadata**: Maintains attributes and coordinates

## xarray Integration

### Data Loading

WindGrib uses xarray as the main backend for data processing:

```python
# Loading with cfgrib
ds = xr.open_dataset(
    grib_file,
    engine='cfgrib',
    decode_timedelta=True,
    backend_kwargs={'errors': 'ignore'},
    chunks={'step': 1, 'latitude': -1, 'longitude': -1}
)
```

### Variable Filtering

For models with many variables, filtering is applied:

```python
# Filter by specific key
ds = xr.open_dataset(
    grib_file,
    engine='cfgrib',
    backend_kwargs={
        'filter_by_keys': {'shortName': 'u10'}
    }
)
```

### Dataset Merging

For subsets with multiple variables:

```python
# Merge multiple datasets
datasets = []
for var_name in ['u10', 'v10']:
    ds_var = xr.open_dataset(..., filter_by_keys={'shortName': var_name})
    datasets.append(ds_var)

# Final merge
merged_ds = xr.merge(datasets, compat='override', join='outer')
```

## Error Handling

### Types of Handled Errors

1. **Files Not Found**: Handle unavailable dates
2. **Failed Downloads**: Retry with backoff
3. **Corrupted Data**: Validate loaded datasets
4. **Missing Variables**: Handle incomplete subsets

### Retry Mechanism

```python
# In find_forecast_time() method
if not idx_files:
    self._retry_count += 1
    if self._retry_count > self.max_retry:
        raise ValueError(f"No files found after {self.max_retry} attempts")
    
    # Search 6h back
    time -= pd.Timedelta(6, 'h')
    new_instance = Grib(time, model=self.model, data_path=self.data_path)
    self.__dict__.update(new_instance.__dict__)
```

## Performance Optimizations

### Parallel Download

Use `ThreadPoolExecutor` to accelerate downloads:

```python
executor = ThreadPoolExecutor(max_workers=100)
download_tasks = [executor.submit(self.download_file, idx_file)
                  for idx_file in idx_files]

# Progress tracking with tqdm
with tqdm(total=len(download_tasks), desc=desc) as progress_bar:
    for _ in as_completed(download_tasks):
        progress_bar.update(1)
```

### Chunked Loading

Memory optimization with chunking:

```python
# Chunking for loading
ds = xr.open_dataset(
    grib_file,
    engine='cfgrib',
    chunks={'step': 1, 'latitude': -1, 'longitude': -1}
)

# Progressive loading
ds.load()  # Load only necessary data
```

### Smart Caching

```python
# Check cache before download
if use_cache and grib_file.exists():
    # Use existing data
    return

# Download only if necessary
self.download_file(idx_file)
```

## Extensibility

### Adding New Models

To add a new model, simply add it to the `MODELS` configuration:

```python
MODELS = {
    'new_model': {
        'product': 'my_product',
        'url': 'https://my-url.com/',
        'key': '{date}/{h:02d}/my_model/',
        'subsets': {
            'temperature': {'variable': ['TMP']},
            'pressure': {'variable': ['PRMSL']}
        }
    }
}
```

### Custom Subsets

You can create custom subsets:

```python
# Create custom subset
custom_config = {
    'my_subset': {
        'variable': ['VAR1', 'VAR2'],
        'step': 0  # Only first time step
    }
}

# Add to GRIB instance
grib._subsets['my_subset'] = GribSubset(
    'my_subset', 
    custom_config['my_subset'], 
    grib
)
```

### Extending Functionality

The `GribSubset` class can be extended to add new functionality:

```python
class MyGribSubset(GribSubset):
    def my_new_method(self):
        # Custom implementation
        ds = self.ds
        # Specific processing
        return result

# Usage
subset = MyGribSubset('name', config, grib_instance)
result = subset.my_new_method()
```

## Development Best Practices

### Memory Management

1. **Always use chunking** for large datasets
2. **Free resources** after use with `ds.close()`
3. **Avoid loading** all time steps into memory

### Download Optimization

1. **Prefer caching** to avoid unnecessary downloads
2. **Use custom paths** to organize data
3. **Limit workers** based on your bandwidth

### Data Validation

1. **Always check** that datasets load correctly
2. **Validate variables** before using them
3. **Handle errors** gracefully

## Conclusion

This technical guide provides a comprehensive overview of WindGrib's architecture and internal mechanisms. For advanced usage or contributing to the project, this deep understanding of components and their interactions is essential.

For more information on the public API and standard usage modes, consult the [Usage Guide](usage_guide.md).