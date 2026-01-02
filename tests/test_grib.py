"""Test cases for the windgrib package."""

import pytest
from windgrib import __version__, __author__, __email__, __license__, __description__


def test_version():
    """Test that version is defined and is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "0.2.7"


def test_metadata():
    """Test that package metadata is correctly defined."""
    assert isinstance(__author__, str)
    assert isinstance(__email__, str)
    assert isinstance(__license__, str)
    assert isinstance(__description__, str)


def test_import():
    """Test that the package can be imported successfully."""
    try:
        import windgrib
        assert windgrib is not None
    except ImportError as e:
        pytest.fail(f"Failed to import windgrib: {e}")


def test_grib_module():
    """Test that the grib module can be imported."""
    try:
        from windgrib import grib
        assert grib is not None
    except ImportError as e:
        pytest.fail(f"Failed to import grib module: {e}")


def test_grib_download():
    from windgrib import Grib
    Grib().download()
# Add more specific tests for your GRIB functionality here
# For example:
# def test_read_grib_file():
#     """Test reading a GRIB file."""
#     from windgrib.grib import read_grib_file
#     # Create a test GRIB file or use a fixture
#     result = read_grib_file("test.grib")
#     assert result is not None