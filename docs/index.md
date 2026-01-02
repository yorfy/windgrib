# WindGrib Documentation

Welcome to the official documentation for WindGrib, a Python library for working with meteorological data in GRIB format.

## 🎯 Project Overview and Goals

WindGrib addresses the specific need for fast, bulk wind data retrieval from meteorological models. While [Herbie](https://github.com/blaylockbk/Herbie) excels at flexible, query-based data access, WindGrib optimizes for a different use case: efficiently downloading complete wind datasets for forecast analysis.

**Design Philosophy:**
- **AWS S3 Focused**: Specifically designed for meteorological data hosted on Amazon S3 (NOAA, ECMWF)
- **s3fs Integration**: Uses s3fs for efficient scanning and discovery of available datasets
- **Bulk Download Optimization**: Designed to quickly retrieve all wind components for a given forecast time
- **Subset-Based Organization**: Pre-configured data subsets (wind, land masks) for immediate use
- **File Consolidation**: Concatenates multiple GRIB files into single datasets per subset, simplifying data management
- **Parallel Processing**: Optimized for concurrent downloads of multiple GRIB files
- **Forecast-Centric**: Focuses on getting complete forecast datasets rather than selective variable queries

**Key Advantage:** While Herbie requires managing multiple individual GRIB files, WindGrib consolidates all related data (e.g., all wind forecast steps) into unified datasets, eliminating the complexity of handling numerous separate files.

**Limitation:** WindGrib is specifically designed for AWS S3-hosted meteorological data and uses s3fs for data discovery. It cannot access data from other sources or protocols.

## 📚 Documentation Structure

- **[Usage Examples](usage_examples.md)** - Practical examples with working code
  - Basic download and data access
  - Data visualization with matplotlib
  - Model comparison (ECMWF vs GFS)
  - Custom models for temperature data

- **[Technical Guide](technical_guide.md)** - Implementation details for developers
  - System architecture and main classes
  - Download mechanisms and performance optimizations
  - xarray integration and extensibility

- **[Custom Models](custom_models.md)** - Guide for extending WindGrib
  - Defining new data models
  - Configuration examples
  - Integration with other data sources

## 🚀 Getting Started

If you're new to WindGrib:

1. **Install the library**:
   ```bash
   pip install windgrib
   ```

2. **Try the basic example** below to understand core concepts

3. **Explore the [usage examples](usage_examples.md)** for specific use cases

4. **Check the [technical guide](technical_guide.md)** for advanced usage

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

## 🔧 Supported Models

WindGrib provides built-in support for wind data from major weather models, with the flexibility to define custom models for additional variables. All data is sourced from Amazon S3 for efficient parallel downloads.

### Default Models (Wind Focus)

| Model | Source | Main Variables | Data Location |
|--------|--------|----------------------|---------------|
| **GFS Wave** | NOAA | UGRD, VGRD (wind components) | Amazon S3 |
| **ECMWF IFS** | ECMWF | 10u, 10v (wind), lsm (land/sea mask) | Amazon S3 |

### Custom Models

WindGrib's architecture allows you to define custom models for additional variables beyond wind data. The [custom models guide](custom_models.md) and [usage examples](usage_examples.md) provide detailed instructions and working examples.

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