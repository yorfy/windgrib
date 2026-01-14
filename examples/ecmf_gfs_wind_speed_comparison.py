"""comparison of wind speed forecast from ECMWF and GFS"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from windgrib import Grib

if __name__ == '__main__':
    print("=== WindGrib Example: ECMWF/GFS Wind Speed Comparison ===\n")
    # ECMWF download
    print("=== ECMWF download ===\n")
    ecmwf_grib = Grib(model='ecmwf_ifs')
    ecmwf_grib.download(clear_cache=True)

    # GFS download
    print("\n=== GFS download ===\n")
    gfs_grib = Grib()
    gfs_grib.step
    gfs_grib.download(clear_cache=True)

    # Get datasets
    ecmwf_wind_ds = ecmwf_grib['wind'].ds
    ecmwf_land_ds = ecmwf_grib['land'].ds
    gfs_ds = gfs_grib['wind'].ds

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
    # Calculate wind speed and convert m/s to knots
    ecmwf_wind_speed = 1.94384 * (ecmwf_wind_ds.u ** 2 + ecmwf_wind_ds.v ** 2) ** 0.5
    ecmwf_wind_speed.attrs['units'] = 'knots'
    ecmwf_wind_speed.attrs['long_name'] = 'Wind Speed'

    # Apply ocean mask to ECMWF wind speed (select time step first to avoid alignment issues)
    ocean_mask = ecmwf_land_ds.lsm.isel(step=0) < 0.5
    ecmwf_wind_speed = ecmwf_wind_speed.where(ocean_mask)
    ecmwf_wind_speed.attrs['long_name'] = 'Wind Speed on ocean'
    print("Applied ocean mask to ECMWF wind speed")

    gfs_wind_speed = 1.94384 * (gfs_ds.u ** 2 + gfs_ds.v ** 2) ** 0.5
    gfs_wind_speed.attrs['units'] = 'knots'
    gfs_wind_speed.attrs['long_name'] = 'Wind Speed'

    ecmwf_wind_speed_now = ecmwf_wind_speed.sel(step=ecmwf_grib.step, method='nearest')
    gfs_wind_speed_now = gfs_wind_speed.sel(step=gfs_grib.step, method='nearest')

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ecmwf_wind_speed_now.plot(ax=ax1, cmap='viridis', vmin=0, vmax=40)
    ax1.set_title(f'ECMWF Wind Speed (Ocean Only)\n{ecmwf_grib.step}')

    gfs_wind_speed_now.plot(ax=ax2, cmap='viridis', vmin=0, vmax=40)
    ax2.set_title(f'GFS Wind Speed\n{gfs_grib.step}')
    plt.tight_layout()
    plt.savefig('wind_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nComparison plot completed with ocean masking! (Wind speeds in knots)")
