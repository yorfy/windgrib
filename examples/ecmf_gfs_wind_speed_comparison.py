"""comparison of wind speed forecast from ECMWF and GFS"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from windgrib import Grib

# ECMWF download
print("=== ECMWF download ===\n")
grib_ecmwf = Grib(model='ecmwf_ifs')
grib_ecmwf.download()

# GFS download
print("\n=== GFS download ===\n")
grib_gfs = Grib()
grib_gfs.download()

# Get datasets
ecmwf_wind_ds = grib_ecmwf['wind']
ecmwf_land_ds = grib_ecmwf['land']
gfs_ds = grib_gfs['wind']

# Convert GFS longitude from 0-360 to -180-180
if 'longitude' in gfs_ds.coords:
    gfs_ds = gfs_ds.assign_coords(longitude=((gfs_ds.longitude + 180) % 360) - 180)
    gfs_ds = gfs_ds.sortby('longitude')

print(f"\nECMWF wind variables: {list(ecmwf_wind_ds.data_vars)}")
print(f"ECMWF wind dimensions: {dict(ecmwf_wind_ds.sizes)}")
if ecmwf_land_ds:
    print(f"ECMWF land variables: {list(ecmwf_land_ds.data_vars)}")
print(f"\nGFS loaded variables: {list(gfs_ds.data_vars)}")
print(f"GFS dimensions: {dict(gfs_ds.sizes)}")

# Wind speed comparison
print("\n=== Wind Speed Comparison ===\n")

# Find common valid times
ecmwf_valid_times = pd.to_datetime(ecmwf_wind_ds.valid_time.values)
gfs_valid_times = pd.to_datetime(gfs_ds.valid_time.values)
common_times = ecmwf_valid_times.intersection(gfs_valid_times)
current_time = pd.Timestamp.utcnow().tz_localize(None)
closest_common_idx = np.abs(common_times - current_time).argmin()
closest_common_time = common_times[closest_common_idx]
ecmwf_step = list(ecmwf_valid_times).index(closest_common_time)
gfs_step = list(gfs_valid_times).index(closest_common_time)
print(f"Using common time closest to now ({current_time}):")
print(f"Common time: {closest_common_time}")
print(f"ECMWF step: {ecmwf_step}, GFS step: {gfs_step}")

# Calculate wind speed and convert m/s to knots
ecmwf_wind_speed = 1.94384 * (ecmwf_wind_ds.u ** 2 + ecmwf_wind_ds.v ** 2) ** 0.5
ecmwf_wind_speed.attrs['units'] = 'knots'
ecmwf_wind_speed.attrs['long_name'] = 'Wind Speed'

gfs_wind_speed = 1.94384 * (gfs_ds.u ** 2 + gfs_ds.v ** 2) ** 0.5
gfs_wind_speed.attrs['units'] = 'knots'
gfs_wind_speed.attrs['long_name'] = 'Wind Speed'

# Apply ocean mask to ECMWF wind speed
lsm = ecmwf_land_ds.lsm
ocean_mask = lsm < 0.5
ecmwf_wind_speed = ecmwf_wind_speed.where(ocean_mask)
print("Applied ocean mask to ECMWF wind speed")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ecmwf_wind_speed.isel(step=ecmwf_step).plot(ax=ax1, cmap='viridis', vmin=0, vmax=40)
ax1.set_title(f'ECMWF Wind Speed (Ocean Only)\n{ecmwf_valid_times[ecmwf_step]}')

gfs_wind_speed.isel(step=gfs_step).plot(ax=ax2, cmap='viridis', vmin=0, vmax=40)
ax2.set_title(f'GFS Wind Speed\n{gfs_valid_times[gfs_step]}')
plt.tight_layout()
plt.savefig('wind_speed_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nComparison plot completed with ocean masking! (Wind speeds in knots)")
