"""Test cases for the Grib class"""
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from windgrib import __version__
from windgrib.grib import Grib, GribSubset

data_dir = Path(__file__).parent / "data"

gb = Grib(data_path=data_dir)


def test_version():
    """Test that version is defined and is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "1.0.0"


def test_grib_getitem():
    """Test Grib __getitem__ method."""

    # Test accessing subset by name
    assert isinstance(gb['wind'], GribSubset)

    # Test accessing attribute
    assert isinstance(gb['step'], np.ndarray)

    # Test invalid key
    with pytest.raises(KeyError, match="'invalid' not found in Grib\\['gfswave'\\]"):
        _ = gb['invalid']

    # Test non-string key
    with pytest.raises(KeyError, match="should be a string"):
        _ = gb[123]


def test_grib_download():
    gb.clear_cache()
    gb['wind'][gb.step[0:10]].download()
    gb_ecmwf = Grib(model='ecmwf_ifs', data_path=data_dir)
    gb_ecmwf.clear_cache()
    gb_ecmwf['wind'][gb_ecmwf.step[0:10]].download()


def test_grib_sel():
    """Test Grib.sel method with different arguments."""

    # Test with custom name
    subset1 = gb.sel(name='custom', var=['UGRD'])
    assert subset1.name == 'custom'
    assert subset1.var == ['UGRD']

    # Test with var only
    subset2 = gb.sel(var=['UGRD'])
    assert subset2.name == "(['UGRD'],,{})"
    assert subset2.var == ['UGRD']

    # Test with step only
    subset3 = gb.sel(step=[0, 1, 2])
    assert subset3.name == "(,[0, 1, 2],{})"
    assert np.all(subset3.step == [0, 1, 2])

    # Test with var and step
    subset4 = gb.sel(var=['UGRD', 'VGRD'], step=[0, 1])
    assert subset4.name == "(['UGRD', 'VGRD'],[0, 1],{})"
    assert subset4.var == ['UGRD', 'VGRD']
    assert np.all(subset4.step == [0, 1])


def test_grib_str():
    """Test Grib __str__ method."""
    assert str(gb) == "Grib['gfswave']"


def test_grib_iter():
    """Test Grib __iter__ method."""
    subsets = list(gb)
    assert 'wind' in subsets


def test_grib_folder_path():
    """Test Grib folder_path property."""
    assert gb.folder_path.exists() or not gb.use_cache


def test_grib_url():
    """Test Grib url property."""
    assert 'noaa-gfs-bdp-pds.s3.amazonaws.com' in gb.url


def test_grib_s3():
    """Test Grib s3 property."""
    assert gb.s3.startswith('s3://')


def test_grib_idx():
    """Test Grib idx property returns DataFrame."""
    assert isinstance(gb.idx, pd.DataFrame)
    assert not gb.idx.empty


def test_grib_step():
    """Test Grib step property."""
    assert isinstance(gb.step, np.ndarray)
    assert len(gb.step) > 0


def test_grib_current_step():
    """Test Grib current_step property."""
    assert isinstance(gb.current_step, (int, np.integer))
    assert gb.current_step in gb.step


def test_grib_subset_str():
    gb_subset = GribSubset('wind', gb, var=['10u', '10v'], step=np.arange(20))
    assert str(gb_subset) == "(wind) ['10u', '10v'] len(step)=20"


def test_grib_subset_from_model():
    """Test GribSubset.from_model with all possible configurations."""
    # Test with str: 'u'
    subset0 = GribSubset.from_model('test1', gb, 'u')
    assert subset0.var == ['u']
    assert np.all(subset0.step == gb.step)
    assert isinstance(subset0.filter_keys, dict) and not subset0.filter_keys

    # Test with list: ['var1', 'var2']
    subset1 = GribSubset.from_model('test1', gb, ['u', 'v'])
    assert subset1.var == ['u', 'v']
    assert np.all(subset1.step == gb.step)
    assert isinstance(subset1.filter_keys, dict) and not subset1.filter_keys

    # Test with tuple: (['var1', 'var2'], step)
    subset2 = GribSubset.from_model('test2', gb, (['u', 'v'], [0, 1, 2]))
    assert subset2.var == ['u', 'v']
    assert np.all(subset2.step == [0, 1, 2])
    assert isinstance(subset2.filter_keys, dict) and not subset2.filter_keys

    # Test with tuple: (['var1', 'var2'], filter_keys)
    subset3 = GribSubset.from_model('test3', gb, (['u', 'v'], {'level': 10}))
    assert subset3.var == ['u', 'v']
    assert np.all(subset3.step == gb.step)
    assert subset3.filter_keys == {'level': 10}

    # Test with tuple: (['var1', 'var2'], step, filter_keys)
    subset4 = GribSubset.from_model('test4', gb, (['u', 'v'], [0, 1], {'level': 10}))
    assert subset4.var == ['u', 'v']
    assert np.all(subset4.step == [0, 1])
    assert subset4.filter_keys == {'level': 10}

    # Test with invalid configuration
    with pytest.raises(TypeError):
        GribSubset.from_model('test5', gb, ('u', {'a': '1'}, [], 'invalid'))


def test_grib_subset_getitem():
    """Test GribSubset __getitem__ method."""
    subset = gb['wind']

    # Test single step selection
    subset_single = subset[0]
    assert subset_single.step == [0]
    assert subset_single.var == subset.var

    # Test multiple steps selection
    subset_multi = subset[[0, 1, 2]]
    assert np.array_equal(subset_multi.step, [0, 1, 2])
    assert subset_multi.var == subset.var

    # Test boolean indexing
    mask = np.array(subset.step) >= subset.current_step()
    subset_next_step = subset[mask]
    assert all(s >= subset.current_step() for s in subset_next_step.step)
    assert subset_next_step.var == subset.var

    # Test invalid step
    with pytest.raises(KeyError, match="not found in step"):
        _ = subset[99999]

    # Test non-integer indexing
    with pytest.raises(KeyError, match="support only integer dtype"):
        _ = subset['invalid']


def test_grib_subset_properties():
    """Test GribSubset file path properties."""
    subset = gb['wind']
    assert subset.grib_file.suffix == '.grib2'
    assert subset.netcdf_file.suffix == '.nc'


def test_grib_subset_idx():
    """Test GribSubset idx property."""
    subset = gb['wind']
    idx = subset.idx
    assert isinstance(idx, pd.DataFrame)
    assert 'step' in idx.columns
    assert 'message_id' in idx.columns


def test_grib_subset_step_property():
    """Test GribSubset step property."""
    subset = gb['wind']
    assert isinstance(subset.step, (np.ndarray, int))
    assert len(subset.step) > 0


def test_grib_subset_current_step():
    """Test GribSubset current_step method."""
    subset = gb['wind']
    assert subset.current_step() == gb.current_step


def test_grib_subset_filter_keys():
    """Test GribSubset with filter_keys."""
    subset = GribSubset('test', gb, var=['UGRD'], step=[0, 1], level=10)
    assert subset.filter_keys == {'level': 10}
    assert subset.var == ['UGRD']
    assert subset.step == [0, 1]
