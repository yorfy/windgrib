"""Benchmark script comparing WindGrib and Herbie performance"""
import time
import xarray
from contextlib import contextmanager
from herbie import FastHerbie, HerbieLatest
from tqdm.dask import TqdmCallback
from windgrib import Grib
import pandas as pd
import sys


@contextmanager
def timer(name, phase_results):
    """Context manager for timing operations."""
    print(f"\n=== {name} ===")
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        phase_results[name.lower().replace(' ', '_')] = duration
        print(f"✓ Completed in {duration:.2f} seconds")


def benchmark_windgrib(use_cache=False):
    """Benchmark WindGrib with or without cache."""
    results = {}

    with timer("init and find forecast", results):
        gb = Grib(model='gfswave')

    if not use_cache:
        gb.clear_cache()

    with timer("download", results):
        gb.download()

    with timer("load", results):
        wind_data = gb['wind'].ds
        gb.to_netcdf()

    return results, wind_data


def benchmark_herbie(use_cache=False):
    """Benchmark Herbie with or without cache."""
    results = {}

    with timer("init and find forecast", results):
        latest = HerbieLatest(model='gfs_wave', product='global.0p25')

        fh_range = list(range(0, 121, 1)) + list(range(123, 385, 3))
        fh = FastHerbie(
            DATES=[latest.date],
            model='gfs_wave',
            product='global.0p25',
            fxx=fh_range,
            max_threads=50,
            priority=['aws']
        )

    with timer("download", results):
        grib_files = fh.download('UGRD|VGRD', overwrite=not use_cache)

    with timer("load", results):
        wind_data = xarray.open_mfdataset(
            grib_files, engine='cfgrib', concat_dim='step',
            combine='nested', coords='minimal', compat='override'
        )
        with TqdmCallback(desc='loading dataset'):
            wind_data.load()

    return results, wind_data


def generate_report(windgrib_no_cache, windgrib_cache, herbie_no_cache, herbie_cache):
    """Generate benchmark report."""
    print("\n# BENCHMARK REPORT: WindGrib vs Herbie")

    # Create DataFrame directly
    df = pd.DataFrame({
        'WindGrib (no cache)': windgrib_no_cache,
        'WindGrib (cache)': windgrib_cache,
        'Herbie (no cache)': herbie_no_cache,
        'Herbie (cache)': herbie_cache
    }).fillna(0)

    # Add total row
    df.loc['total'] = df.sum()

    print("\n## Detailed Phase Timings (seconds)")
    print("\n```")
    print(df.to_string(float_format='%.2f'))
    print("```")

    # Phase-by-phase analysis
    print("\n## Phase-by-Phase Analysis")

    # Compare phases present in both
    for phase in df.index[:-1]:  # exclude 'total'
        wg_time = df.loc[phase, 'WindGrib (no cache)']
        h_time = df.loc[phase, 'Herbie (no cache)']

        if wg_time > 0 and h_time > 0:
            ratio = wg_time / h_time
            comp = f"WindGrib {1/ratio:.1f}x faster" if ratio < 1 else f"WindGrib {ratio:.1f}x slower"
            print(f"\n- **{phase.replace('_', ' ').title()}**: {comp} ({wg_time:.2f}s vs {h_time:.2f}s)")

    # Cache performance
    print("\n## Cache Performance")

    wg_speedup = df.loc['total', 'WindGrib (no cache)'] / df.loc['total', 'WindGrib (cache)']
    h_speedup = df.loc['total', 'Herbie (no cache)'] / df.loc['total', 'Herbie (cache)']

    print(f"\n- WindGrib cache speedup: **{wg_speedup:.1f}x**")
    print(f"- Herbie cache speedup: **{h_speedup:.1f}x**")

    # Overall comparison
    print("\n## Overall Comparison")

    total_wg = df.loc['total', 'WindGrib (no cache)']
    total_h = df.loc['total', 'Herbie (no cache)']

    if total_h > 0:
        ratio = total_wg / total_h
        comparison = f"WindGrib is **{1/ratio:.1f}x faster**" if ratio < 1 else f"WindGrib is **{ratio:.1f}x slower**"
        print(f"\n{comparison} overall ({total_wg:.2f}s vs {total_h:.2f}s)")


def main():
    """Run complete benchmark."""
    print("# WindGrib vs Herbie Benchmark")
    print("\nTesting both libraries with and without cache")

    # Run benchmarks
    print("\n## Phase 1: Testing without cache (first run)")

    windgrib_no_cache, _ = benchmark_windgrib(use_cache=False)
    herbie_no_cache, _ = benchmark_herbie(use_cache=False)

    print("\n## Phase 2: Testing with cache (second run)")

    windgrib_cache, _ = benchmark_windgrib(use_cache=True)
    herbie_cache, _ = benchmark_herbie(use_cache=True)

    # Generate report
    generate_report(windgrib_no_cache, windgrib_cache, herbie_no_cache, herbie_cache)

    print("\n---\n*Benchmark completed successfully!*")


if __name__ == '__main__':
    # Force UTF-8 encoding for output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
