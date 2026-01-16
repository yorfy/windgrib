"""
WindGrib - A Python package for working with GRIB weather data files.

This package provides tools to read, parse, and analyze GRIB format weather data.
"""

__version__ = "0.2.7"
__author__ = "yorfy"
__license__ = "MIT"
__description__ = "A Python package for working with GRIB weather data files"

# Import main class
from .grib import Grib

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "Grib",
    "grib_to_dataset"
]
