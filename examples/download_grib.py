"""Minimal example of downloading and accessing data,
using cache and saving data to netcdf format to speedup further data reading"""

from windgrib import Grib

# Create a GRIB instance for the GFS Wave model
grib = Grib(model='gfswave')

# Download the data
print("Downloading GFS Wave data...")
grib.download(use_cache=True)
# use_cache=True is the default option
# But you can use use_cache=False to force downloading ignoring cache files

# Access wind data
wind_data = grib['wind']

# Display basic information
print(f"Available variables: {list(wind_data.data_vars)}")
print(f"Dimensions: {dict(wind_data.sizes)}")
print(f"Time period: {wind_data.time.values}")

# Access specific subset
print(f"U data shape: {wind_data.u.shape}")
print(f"V data shape: {wind_data.v.shape}")

# save to netcdf file to speedup further data reading
grib.to_nc()