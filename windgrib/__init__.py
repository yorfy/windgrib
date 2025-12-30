"""
WindGrib - A Python package for working with GRIB weather data files.

This package provides tools to read, parse, and analyze GRIB format weather data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
__description__ = "A Python package for working with GRIB weather data files"

# Import main functionality
from .grib import *  # This will expose all functions from grib.py at package level

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__description__"
]