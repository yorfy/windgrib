"""Test cases for the Grib class"""
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from windgrib import __version__
from windgrib.grib import Grib, GribSubset, MODELS

data_dir = Path(__file__).parent / "data"

gb = Grib(data_path=data_dir)


def test_version():
    """Test that version is defined and is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "1.0.1"


def test_grib_getitem():
    """Test Grib __getitem__ method."""

    # Test accessing subset by name
    assert isinstance(gb['wind'], GribSubset)

    # Test accessing subset by attribute name
    assert isinstance(gb.wind, GribSubset)

    # Test accessing attribute
    assert isinstance(gb['step'], np.ndarray)
    assert np.all(gb.step == gb['step'])

    # Test invalid key
    with pytest.raises(KeyError):
        _ = gb['invalid']
    with pytest.raises(AttributeError):
        _ = gb.invalid

    # Test integer indexing - single step
    gb_single = gb[0]
    assert isinstance(gb_single, Grib)
    assert len(gb_single.step) == 1
    assert gb_single.step[0] == gb.step[0]
    assert gb_single is not gb

    # Test integer array indexing - multiple steps
    gb_multi = gb[[0, 1, 2]]
    assert isinstance(gb_multi, Grib)
    assert len(gb_multi.step) == 3
    assert np.array_equal(gb_multi.step, gb.step[[0, 1, 2]])
    assert gb_multi is not gb

    # Test slice indexing
    gb_slice = gb[0:5]
    assert isinstance(gb_slice, Grib)
    assert len(gb_slice.step) == 5
    assert np.array_equal(gb_slice.step, gb.step[0:5])
    assert gb_slice is not gb

    # Test boolean indexing
    mask = gb.step < 10
    gb_bool = gb[mask]
    assert isinstance(gb_bool, Grib)
    assert np.all(gb_bool.step < 10)
    assert gb_bool is not gb

    # Test negative indexing
    gb_last_step = gb[-1]
    assert isinstance(gb_last_step, Grib)
    assert len(gb_last_step.step) == 1
    assert gb_last_step.step[0] == gb.step[-1]

    # Test that original gb is not modified
    original_step = gb.step.copy()
    _ = gb[0:5]
    assert np.array_equal(gb.step, original_step)


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

    # Test with valid filter key from idx columns
    subset5 = gb.sel(var=['UGRD'], layer='surface')
    assert subset5.filter_keys == {'layer': 'surface'}

    # Test with invalid filter key
    with pytest.raises(KeyError):
        gb.sel(var=['UGRD'], invalid_key='value')


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
    assert isinstance(subset_single, GribSubset)
    assert len(subset_single.step) == 1
    assert subset_single.step[0] == subset.step[0]
    assert subset_single.var == subset.var
    assert subset_single is not subset

    # Test multiple steps selection
    subset_multi = subset[[0, 1, 2]]
    assert isinstance(subset_multi, GribSubset)
    assert len(subset_multi.step) == 3
    assert np.array_equal(subset_multi.step, subset.step[[0, 1, 2]])
    assert subset_multi.var == subset.var
    assert subset_multi is not subset

    # Test slice indexing
    subset_slice = subset[0:5]
    assert isinstance(subset_slice, GribSubset)
    assert len(subset_slice.step) == 5
    assert np.array_equal(subset_slice.step, subset.step[0:5])
    assert subset_slice.var == subset.var
    assert subset_slice is not subset

    # Test boolean indexing
    mask = np.array(subset.step) >= subset.current_step()
    subset_bool = subset[mask]
    assert isinstance(subset_bool, GribSubset)
    assert all(s >= subset.current_step() for s in subset_bool.step)
    assert subset_bool.var == subset.var
    assert subset_bool is not subset

    # Test negative indexing
    subset_neg = subset[-1]
    assert isinstance(subset_neg, GribSubset)
    assert len(subset_neg.step) == 1
    assert subset_neg.step[0] == subset.step[-1]

    # Test that original subset is not modified
    original_step = subset.step.copy()
    _ = subset[0:5]
    assert np.array_equal(subset.step, original_step)


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
    assert np.all(subset.step == [0, 1])


def test_grib_subset_sel():
    """Test GribSubset.sel method."""
    subset = Grib(model='ecmwf_ifs')['wind']

    # Test with only one step (keeps parent name by default)
    subset_step = subset.sel(step=3)
    assert isinstance(subset_step, GribSubset)
    assert subset_step.name == 'wind'
    assert subset_step.var == subset.var
    assert np.array_equal(subset_step.step, [3])

    # Test with custom name and only two steps
    subset_named = subset.sel(name='two_first_steps', step=[3, 6])
    assert subset_named.name == 'two_first_steps'
    assert subset_named.var == subset.var
    assert np.array_equal(subset_named.step, [3, 6])

    # Test that original subset is not modified
    original_filter_keys = subset.filter_keys.copy()
    original_step = subset.step.copy()
    new_subset = subset.sel(step=[0, 12, 24])
    assert np.array_equal(new_subset.step, [0, 12, 24])
    assert new_subset.filter_keys == original_filter_keys
    assert subset.filter_keys == original_filter_keys
    assert np.array_equal(subset.step, original_step)

    # Test with invalid filter key
    with pytest.raises(KeyError):
        subset.sel(invalid_key='value')

def test_several_level():
    """Test GribSubset with several level."""
    MODELS['ecmwf_t'] = {
        'product': 'oper',
        'url': 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/',
        'key': '{date}/{h:02d}z/ifs/0p25/oper/',
        'idx': '.index',
        'subsets': {'temperature': 't'}
    }
    grib_temperature = Grib(model='ecmwf_t', data_path=data_dir)
    subset = grib_temperature['temperature']
    subset = subset.sel(levelist = ['50', '100', '200'])
    subset.download()
    ds = subset.ds
    ds = ds.assign_coords(step_original=ds.step)
    ds = ds.set_index(step=['step_original', 'isobaricInhPa']).unstack('step')
    ds = ds.rename({'step_original': 'step'})
    print(ds)
