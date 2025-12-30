# WindGrib 🌬️

[![PyPI version](https://badge.fury.io/py/windgrib.svg)](https://badge.fury.io/py/windgrib)
[![Python Version](https://img.shields.io/pypi/pyversions/windgrib)](https://pypi.org/project/windgrib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for reading, parsing, and analyzing GRIB weather data files.

## Features

- ✅ Read GRIB weather data files
- ✅ Parse meteorological data
- ✅ Easy-to-use Python interface
- ✅ Modern Python packaging
- ✅ Type hints support
- ✅ Comprehensive error handling

## Installation

### From PyPI (when published)

```bash
pip install windgrib
```

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/windgrib.git
cd windgrib

# Install the package
pip install .

# Or install in development mode
pip install -e .
```

### Development setup

```bash
# Clone the repository
git clone https://github.com/yourusername/windgrib.git
cd windgrib

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Or install main package + dev extras
pip install -e ".[dev]"
```

## Usage

```python
from windgrib import read_grib_file

# Read a GRIB file
data = read_grib_file("weather.grib")

# Access weather data
print(f"Temperature: {data['temperature']}°C")
print(f"Wind speed: {data['wind_speed']} km/h")
```

## Example

Check out the [example](examples/windgrib_example.py) for a complete usage demonstration.

## Development

### Install development dependencies

```bash
pip install -e .[dev]
```

### Run tests

```bash
pytest
```

### Format code

```bash
black .
isort .
```

### Type checking

```bash
mypy windgrib
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub.

---

*Made with ❤️ for weather data enthusiasts*