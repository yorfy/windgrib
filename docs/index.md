# WindGrib Documentation

Welcome to the official documentation for WindGrib, a Python library for working with meteorological data in GRIB format.

## 🎯 Project Overview and Goals

WindGrib is inspired by [Herbie](https://github.com/blaylockbk/Herbie), a comprehensive meteorological data access library, but focuses specifically on efficient wind data extraction and targeted variable downloading.

**Key Innovations:**
- **Subset-Based Downloads**: Uses GRIB index files to download only the specific variables you need (e.g., wind components UGRD/VGRD), reducing bandwidth and storage requirements
- **Automatic Latest Data**: Automatically retrieves the most recent available forecast data without manual date specification
- **Smart Caching**: Incremental download completion - if data isn't immediately available, the system automatically resumes as new data becomes available

## 📚 Documentation Structure

### For Users

- **[Usage Guide](usage_guide.md)** - Complete guide to different usage modes
  - Installation and configuration
  - GRIB data download
  - Data reading and analysis
  - NetCDF conversion
  - Model comparison
  - Best practices

- **[Usage Examples](usage_examples.md)** - Practical and executable examples
  - Basic examples
  - Meteorological calculations
  - Data visualization
  - Error handling
  - Advanced usage

### For Developers

- **[Technical Guide](technical_guide.md)** - Implementation details
  - System architecture
  - Main classes
  - Download mechanisms
  - xarray integration
  - Performance optimizations
  - Extensibility

## 🚀 Getting Started

If you're new to WindGrib, we recommend starting with:

1. **Install the library**:
   ```bash
   pip install windgrib
   ```

2. **Read the Usage Guide** to understand basic concepts and main usage modes.

3. **Explore the examples** to see concrete use cases.

4. **Consult the technical guide** if you want to contribute or understand implementation details.

## 📈 Quick Example

```python
from windgrib import Grib

# Download GFS wind data (default subset)
grib = Grib(model='gfswave')
grib.download()

# Access wind data
wind_data = grib['wind']

# Calculate wind speed
import numpy as np
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)

print(f"Average speed: {wind_speed.mean().values:.2f} m/s")
```

**Custom Model Example** - Extending to temperature data:

```python
from windgrib.grib import MODELS, Grib

# Define custom model for temperature data
MODELS['gfs_atmos'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {'variable': ['TMP']}
    }
}

# Download temperature data using custom model
grib_temp = Grib(model='gfs_atmos')
grib_temp.download()
temp_data = grib_temp['temperature']
```

## 🔧 Supported Models

WindGrib provides built-in support for wind data from major weather models, with the flexibility to define custom models for additional variables. All data is sourced from Amazon S3 for efficient parallel downloads.

### Default Models (Wind Focus)

| Model | Source | Main Variables | Data Location |
|--------|--------|----------------------|---------------|
| **GFS Wave** | NOAA | UGRD, VGRD (wind components) | Amazon S3 |
| **ECMWF IFS** | ECMWF | 10u, 10v (wind), lsm (land/sea mask) | Amazon S3 |

### Custom Models

WindGrib's architecture allows you to define custom models for additional variables beyond wind data. The [custom models guide](custom_models.md) provides detailed instructions on how to:

- Define new data models for different variables (temperature, pressure, humidity, etc.)
- Configure custom data sources and file structures
- Extend functionality through subclassing
- Integrate with other meteorological data providers

Example custom model for temperature data:
```python
MODELS['my_temp_model'] = {
    'product': 'custom_product',
    'url': 'https://my-s3-bucket.s3.amazonaws.com/',
    'key': '{date}/{h:02d}/custom_path/',
    'subsets': {
        'temperature': {'variable': ['TMP']},
        'humidity': {'variable': ['RH']}
    }
}
```

## 🤝 Contributing

This documentation is open-source and contributions are welcome! If you find errors or want to add examples, feel free to:

- Open an issue on GitHub
- Submit a pull request
- Suggest improvements

## 📬 Support

For any questions or issues:

- Check the [GitHub issues](https://github.com/your-org/windgrib/issues)
- Read our [contribution guide](CONTRIBUTING.md)
- Explore the provided examples

## 📜 License

This documentation is published under the MIT license, just like the WindGrib source code.

---

© 2025 WindGrib. All rights reserved.