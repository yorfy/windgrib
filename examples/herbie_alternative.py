"""
Alternative script using Fast Herbie and HerbieLatest to download GFS wind data
Demonstrates equivalent functionality to WindGrib using Herbie's optimized methods
"""
import xarray
from herbie import FastHerbie, HerbieLatest
from tqdm.dask import TqdmCallback

"""Download latest GFS wind data using Fast Herbie and HerbieLatest."""

print("=== Using Herbie for GFS Wind Data ===")

# Get latest available GFS run
print("1. Finding latest GFS run...")
latest = HerbieLatest(model='gfs_wave', product='global.0p25')
print(f"Latest run: {latest.date}")

# Use FastHerbie for bulk download of wind components
print("2. Setting up FastHerbie for wind data...")

# Create FastHerbie instance for multiple forecast hours
fh_range = list(range(0, 121, 1)) + list(range(123, 385, 3))  # 0 to 360 hours (15 days), every hour

FH = FastHerbie(
    DATES=[latest.date],  # FastHerbie expects a list of dates
    model='gfs_wave',
    product='global.0p25',
    fxx = fh_range,
    max_threads=50
)

# Download wind components (UGRD, VGRD) in one call
print("3. Downloading wind components (UGRD, VGRD)...")

grib_files = FH.download('UGRD|VGRD', verbose=True)

print(f"3. Reading {len(grib_files)} GRIB files")

ds = xarray.open_mfdataset(grib_files, engine='cfgrib', concat_dim='step',
                           combine='nested', coords='minimal', compat='override',
                           )

print("4. loading the dataset")
with TqdmCallback(desc=f'loading dataset'):
    ds.load()