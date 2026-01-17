"""Benchmark script comparing grib_to_dataset vs xarray.open_dataset with cfgrib."""

import time
import tempfile
from pathlib import Path
import xarray as xr
import numpy as np
from windgrib import Grib
from windgrib.grib_to_dataset import grib_to_dataset


def xarray_cfgrib(grib_bytes):
    """Convert GRIB bytes to xarray using cfgrib engine."""
    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp_file:
        tmp_file.write(grib_bytes)
        tmp_path = Path(tmp_file.name)
    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        return ds.load()
    finally:
        tmp_path.unlink(missing_ok=True)


def benchmark(grib_bytes, name, func):
    """Benchmark a function."""
    print(f"\n=== {name} ===")
    start = time.time()
    result = func(grib_bytes)
    duration = time.time() - start
    print(f"Time: {duration:.2f}s")
    print(f"Variables: {list(result.data_vars.keys())}")
    return duration, result


def main():
    print("Benchmark: grib_to_dataset vs xarray.open_dataset with cfgrib")
    print("=" * 60)

    gb = Grib(model='gfswave', use_cache=False)
    gb.download()
    gbs_wind = gb['wind']
    gbs_wind.download()
    grib_bytes = gbs_wind._grib_data

    if not grib_bytes:
        print("No GRIB data available")
        return

    print(f"\nTesting with {len(grib_bytes)} bytes of GRIB data")

    # Benchmark both methods
    t1, ds1 = benchmark(grib_bytes, "grib_to_dataset", grib_to_dataset)
    t2, ds2 = benchmark(grib_bytes, "xarray.open_dataset with cfgrib", xarray_cfgrib)

    # Compare datasets
    print("\n" + "=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)
    print(f"Variables match: {set(ds1.data_vars) == set(ds2.data_vars)}")
    print(f"Dimensions match: {set(ds1.dims) == set(ds2.dims)}")
    for var in ds1.data_vars:
        if var in ds2.data_vars:
            v1 = ds1[var].values
            v2 = ds2[var].values
            print(f"\n{var}:")
            print(f"  Shape: ds1={v1.shape}, ds2={v2.shape}")
            print(f"  Dtype: ds1={v1.dtype}, ds2={v2.dtype}")
            print(f"  NaN count: ds1={np.isnan(v1).sum()}, ds2={np.isnan(v2).sum()}")
            v1_valid = v1[~np.isnan(v1)]
            v2_valid = v2[~np.isnan(v2)]
            if len(v1_valid) > 0 and len(v2_valid) > 0:
                print(f"  Min: ds1={v1_valid.min():.6f}, ds2={v2_valid.min():.6f}")
                print(f"  Max: ds1={v1_valid.max():.6f}, ds2={v2_valid.max():.6f}")
                print(f"  Mean: ds1={v1_valid.mean():.6f}, ds2={v2_valid.mean():.6f}")
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.any():
                    diff = np.abs(v1[mask] - v2[mask])
                    print(f"  Max diff: {diff.max():.6e}")
                    print(f"  Mean diff: {diff.mean():.6e}")
                    print(f"  Values match (rtol=1e-5): {np.allclose(v1[mask], v2[mask], rtol=1e-5)}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    speedup = t2 / t1
    print(f"grib_to_dataset:              {t1:.2f}s")
    print(f"xarray.open_dataset (cfgrib): {t2:.2f}s")
    print(f"Speedup: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
