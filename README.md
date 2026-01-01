# WindGrib

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://pypi.org/project/windgrib/)

A Python library for downloading, reading, and processing meteorological data in GRIB format.

## 🌍 Overview

WindGrib focuses on efficient wind data extraction and targeted variable downloading from meteorological models. Key features include:

- **Subset-Based Downloads**: Download only specific variables using GRIB index files
- **Automatic Latest Data**: Retrieves the most recent available forecast data
- **Smart Caching**: Intelligent caching with incremental download completion
- **Multi-Model Support**: GFS Wave, ECMWF IFS, and custom model definitions

## 🚀 Installation

```bash
pip install windgrib
```

## 📈 Basic Usage

```python
from windgrib import Grib
import numpy as np

# Download GFS wind data
grib = Grib(model='gfswave')
grib.download()

# Access wind data
wind_data = grib['wind']

# Calculate wind speed
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)
print(f"Average speed: {wind_speed.mean().values:.2f} m/s")
```

## 📚 Documentation

For complete documentation, examples, and advanced usage:

**[📖 View Full Documentation](docs/index.md)**

- [Usage Examples](docs/usage_examples.md) - Practical examples with working code
- [Technical Guide](docs/technical_guide.md) - Implementation details for developers
- [Custom Models](docs/custom_models.md) - Guide for extending WindGrib

## 🤝 Contributing

Contributions are welcome! See our [contribution guide](CONTRIBUTING.md) for more information.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

© 2025 WindGrib. All rights reserved.