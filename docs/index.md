# WindGrib Documentation

Welcome to the official documentation for WindGrib, a Python library for working with meteorological data in GRIB format.

## 🎯 Project Overview and Goals

### Origin and Motivation

WindGrib was born from a practical need in sailing weather routing: efficiently retrieving meteorological data as soon as it becomes available. Weather routing software for sailboats requires complete, up-to-date wind forecasts to calculate optimal routes. The challenge was to automatically download the latest forecast data from multiple sources (NOAA GFS, ECMWF) the moment they're published, without manual intervention or complex file management.

While [Herbie](https://github.com/blaylockbk/Herbie) excels at flexible, query-based data access ([see equivalent Herbie implementation](../examples/herbie_alternative.py)), WindGrib optimizes for this specific use case: efficiently downloading complete wind datasets for immediate use in routing applications. **[Benchmark results](benchmark_results.md)** show WindGrib is 2.9x faster than Herbie overall, making it ideal for time-sensitive applications where getting the latest forecast quickly matters.

**Design Philosophy:**
- **AWS S3 Focused**: Specifically designed for meteorological data hosted on Amazon S3 (NOAA, ECMWF)
- **Automatic Latest Data**: Detects and downloads the most recent available forecast without manual date specification
- **Asyncio-Based Downloads**: Concurrent downloads using asyncio (2.3x faster than FastHerbie's multi-threading)
- **Parallel GRIB Decoding**: Process-based parallel decoding (2.6x faster, cfgrib doesn't support parallel reading on Windows)
- **Incremental NetCDF Caching**: Smart caching provides 6.5x speedup on subsequent runs
- **Subset-Based Organization**: Pre-configured data subsets (wind, land masks) for immediate use
- **File Consolidation**: Concatenates multiple GRIB files into single datasets per subset, simplifying data management
- **Forecast-Centric**: Focuses on getting complete forecast datasets rather than selective variable queries

**Key Advantage:** While Herbie requires managing multiple individual GRIB files, WindGrib consolidates all related data (e.g., all wind forecast steps) into unified datasets, eliminating the complexity of handling numerous separate files—critical for automated weather routing workflows.

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

- **[Benchmark Results](benchmark_results.md)** - Performance comparison with Herbie
  - Detailed timing analysis
  - Asyncio vs multi-threading comparison
  - Cache performance evaluation

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
gb = Grib(model='gfswave')
gb.download()

# Access wind data
wind_data = gb['wind'].ds

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
