"""Test cases for the windgrib package."""
from logging import raiseExceptions

import numpy as np
import pandas as pd
import pytest
from windgrib import __version__, __author__, __license__, __description__, Grib
from windgrib.grib import GribSubset


def test_version():
    """Test that version is defined and is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "0.2.7"


def test_grib_subset_str():
    gb = Grib()
    gb_subset = GribSubset('wind', gb, ['10u', '10v'], np.arange(20))
    assert str(gb_subset) == "(wind) ['10u', '10v'] len(step)=20"


def test_grib_subset_getitem():
    """Test that the grib subset can be retrieved or created successfully."""
    gb = Grib(timestamp=pd.Timestamp.utcnow(), use_cache=False)
    assert isinstance(gb['wind'], GribSubset)
    with pytest.raises(KeyError, match="'invalid_name' not found in Grib\['gfswave'\]"):
        _ = gb['invalid_name']
    assert np.all(gb['wind'].step == gb['step'])
    assert np.all(gb['wind', [0, 1, 2]].step == [0, 1, 2])


def test_grib_download():
    from windgrib import Grib
    gb = Grib()
    gb.download()


# Add more specific tests for your GRIB functionality here
# For example:
# def test_read_grib_file():
#     """Test reading a GRIB file."""
#     from windgrib.grib import read_grib_file
#     # Create a test GRIB file or use a fixture
#     result = read_grib_file("test.grib")
#     assert result is not None
