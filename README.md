# WindGrib

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://pypi.org/project/windgrib/)

A Python library for downloading, reading, and processing meteorological data in GRIB format.

## 🌍 Overview

WindGrib is a powerful library for working with meteorological data in GRIB (GRIdded Binary) format. Inspired by [Herbie](https://github.com/blaylockbk/Herbie), which provides access to a wide range of meteorological data, WindGrib focuses specifically on wind data extraction and optimization.

**Key Differences from Herbie:**
- **Targeted Data Extraction**: While Herbie offers comprehensive access to various meteorological variables, WindGrib specializes in downloading only the specific variables you need (wind components, temperature, etc.) using GRIB index files for efficient subset selection.
- **Automatic Latest Data Retrieval**: WindGrib automatically fetches the most recent available data, eliminating the need to manually specify dates.
- **Incremental Cache Management**: The library features intelligent caching that allows for gradual data completion. If data isn't immediately available, WindGrib will automatically resume downloads as new data becomes available, ensuring you always have the most complete dataset.

WindGrib primarily focuses on wind data extraction from major weather models, with the flexibility to define custom models for additional variables. The library provides a simple interface for downloading, reading, converting, and analyzing meteorological data. Leveraging Amazon S3 data sources, WindGrib enables parallel downloads for faster data retrieval while maintaining a strong emphasis on efficiency and targeted data extraction.

## 🚀 Installation

```bash
pip install windgrib
```

## 📦 Main Features

- **Automatic Download**: Downloads GRIB data from remote sources (NOAA, ECMWF)
- **Multi-Model Support**: Supports GFS Wave and ECMWF IFS
- **Format Conversion**: Converts GRIB files to NetCDF for more efficient use
- **Easy Data Access**: Simple interface for accessing data subsets
- **xarray Integration**: Works seamlessly with xarray objects for analysis
- **Cache Management**: Avoids unnecessary downloads with intelligent caching system

## 📚 Documentation

For complete documentation on different usage modes, see our [Usage Guide](docs/usage_guide.md).

## 🔧 Main Usage Modes

### 1. GRIB Data Download

```python
from windgrib import Grib

# Download GFS Wave data
grib = Grib(model='gfswave')
grib.download()

# Download ECMWF data with custom options
grib_ecmwf = Grib(
    time='2023-12-25 12:00',
    model='ecmwf_ifs',
    data_path='./my_data'
)
grib_ecmwf.download(use_cache=False)
```

### 2. Data Reading and Analysis

```python
from windgrib import Grib
import numpy as np

# Load data
grib = Grib(model='gfswave')
grib.download()

# Access wind data
wind_data = grib['wind']

# Calculate wind speed
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
print(f"Average speed: {wind_speed.mean().values} m/s")
```

### 3. Conversion to NetCDF

```python
from windgrib import Grib

# Convert data to NetCDF format
grib = Grib(model='ecmwf_ifs')
grib.download()
grib.to_nc()  # Convert all subsets
```

### 4. GFS Atmospheric Temperature Data

```python
from windgrib.grib import MODELS, Grib
import numpy as np

# Define GFS atmospheric model
MODELS['gfs_atmos'] = {
    'product': 'global.0p25',
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': {'variable': ['TMP']}
    }
}

# Download temperature data
grib = Grib(model='gfs_atmos')
grib.download()

# Analyze temperatures
temp_data = grib['temperature']
temp_celsius = temp_data.TMP - 273.15  # Conversion K -> °C

print(f"Average temperature: {temp_celsius.mean().values:.2f}°C")
print(f"Minimum temperature: {temp_celsius.min().values:.2f}°C")
print(f"Maximum temperature: {temp_celsius.max().values:.2f}°C")
```

### 5. Model Comparison

```python
from windgrib import Grib
import numpy as np

# Download and compare ECMWF vs GFS
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_gfs = Grib(model='gfswave')

grib_ecmwf.download()
grib_gfs.download()

# Compare wind speeds
 ecmwf_wind = grib_ecmwf['wind']
 gfs_wind = grib_gfs['wind']

 ecmwf_speed = (ecmwf_wind.u**2 + ecmwf_wind.v**2)**0.5
 gfs_speed = (gfs_wind.u**2 + gfs_wind.v**2)**0.5

difference = ecmwf_speed - gfs_speed
print(f"Average difference: {difference.mean().values}")
```

## 📈 Complete Examples

- **Wind Comparison**: See our [comparison example](examples/ecmf_gfs_wind_speed_comparison.py) between ECMWF and GFS models for wind data
- **GFS Temperature Data**: See the [temperature example](examples/temperature_variation_near_toulouse.py) demonstrating how to define and use custom models beyond the default wind subsets (GFS and ECMWF). This example shows how to extend WindGrib's capabilities to download and analyze different atmospheric variables like temperature data.

## 🔧 Configuration

### Supported Models

| Model | Identifier | Main Variables |
|--------|-------------|----------------------|
| GFS Wave | `gfswave` | UGRD, VGRD (wind components) |
| ECMWF IFS | `ecmwf_ifs` | 10u, 10v (wind), lsm (land/sea mask) |
| GFS Atmospheric | `gfs_atmos` | TMP (temperature) - *custom model* |

### Configuration Options

- `model`: Weather model to use (`'gfswave'` or `'ecmwf_ifs'`)
- `time`: Date and time for data (default: now)
- `data_path`: Path to store data (default: `./data/grib`)
- `max_retry`: Maximum number of download attempts (default: 10)

## 🧪 Tests

```bash
pytest tests/
```

## 🤝 Contribution

Contributions are welcome! See our [contribution guide](CONTRIBUTING.md) for more information.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📬 Contact

For any questions or suggestions, feel free to open an issue on GitHub.

---

© 2025 WindGrib. All rights reserved.