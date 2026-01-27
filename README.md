# WindGrib

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.1-orange.svg)](https://pypi.org/project/windgrib/)

A Python library for downloading, reading, and processing meteorological data in GRIB format.

## 🌍 Overview

WindGrib focuses on efficient wind data extraction and targeted variable downloading from meteorological models. Key features include:

- **Subset-Based Downloads**: Download only specific variables using GRIB index files
- **Automatic Latest Data**: Retrieves the most recent available forecast data
- **Smart Caching**: Intelligent caching with incremental download completion
- **Multi-Model Support**: GFS Wave, ECMWF IFS, and custom model definitions
- **AWS S3 Focused**: Specifically designed for meteorological data hosted on Amazon S3
- **High Performance**: Asyncio-based downloads and parallel GRIB decoding for maximum speed

## 🚀 Installation

```bash
pip install windgrib
```

## 📈 Basic Usage

```python
from windgrib import Grib
import numpy as np

# Download GFS wind data
gb = Grib(time='2026/01/3', model='gfswave')
gb.download()

# Access wind data
wind_data = gb['wind'].ds

# Calculate wind speed
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
print(f"Average speed: {wind_speed.mean().values:.2f} m/s")
```

## 📚 Documentation

For complete documentation, examples, and advanced usage:

**[📖 View Full Documentation](https://github.com/yorfy/windgrib/blob/master/docs/index.md)**

- [Usage Examples](https://github.com/yorfy/windgrib/tree/master/docs/usage_examples.md) - Practical examples with working code
- [Technical Guide](https://github.com/yorfy/windgrib/tree/master/docs/technical_guide.md) - Implementation details for developers
- [Custom Models](https://github.com/yorfy/windgrib/tree/master/docs/custom_models.md) - Guide for extending WindGrib
- [Benchmark Results](https://github.com/yorfy/windgrib/tree/master/docs/benchmark_results.md) - Performance comparison with Herbie

## 🤝 Contributing

Contributions are welcome! See our [contribution guide](https://github.com/yorfy/windgrib/blob/master/CONTRIBUTING.md) for more information.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yorfy/windgrib/blob/master/LICENSE) file for more details.

---

© 2025 WindGrib. All rights reserved.
