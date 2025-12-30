import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from windgrib import Grib


def ecmf_gfs_wind_speed_comparison():
    # comparison of wind speed forecast from ECMWF and GFS

    # ECMWF download
    print("=== ECMWF download ===\n")
    grib_ecmwf = Grib(model='ecmwf_ifs')
    grib_ecmwf.download()
    grib_ecmwf.to_nc()

    # GFS download
    print("\n=== GFS download ===\n")
    grib_gfs = Grib()
    grib_gfs.download()
    grib_gfs.to_nc()

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

    if len(common_times) == 0:
        print("No common valid times found between ECMWF and GFS")
        current_time = pd.Timestamp.now()
        ecmwf_closest_idx = np.abs(ecmwf_valid_times - current_time).argmin()
        gfs_closest_idx = np.abs(gfs_valid_times - current_time).argmin()
        ecmwf_time = ecmwf_valid_times[ecmwf_closest_idx]
        gfs_time = gfs_valid_times[gfs_closest_idx]
        print(f"Using closest times to now ({current_time}):")
        print(f"ECMWF: {ecmwf_time} (step={ecmwf_closest_idx})")
        print(f"GFS: {gfs_time} (step={gfs_closest_idx})")
        ecmwf_step = ecmwf_closest_idx
        gfs_step = gfs_closest_idx
    else:
        current_time = pd.Timestamp.now()
        closest_common_idx = np.abs(common_times - current_time).argmin()
        closest_common_time = common_times[closest_common_idx]
        ecmwf_step = list(ecmwf_valid_times).index(closest_common_time)
        gfs_step = list(gfs_valid_times).index(closest_common_time)
        print(f"Using common time closest to now ({current_time}):")
        print(f"Common time: {closest_common_time}")
        print(f"ECMWF step: {ecmwf_step}, GFS step: {gfs_step}")

    # Calculate wind speed in m/s then convert to knots
    ecmwf_speed = (ecmwf_wind_ds.u ** 2 + ecmwf_wind_ds.v ** 2) ** 0.5
    ecmwf_speed_knots = ecmwf_speed * 1.94384  # Convert m/s to knots
    ecmwf_speed_knots.attrs['units'] = 'knots'
    ecmwf_speed_knots.attrs['long_name'] = 'Wind Speed'

    if 'v' in gfs_ds.data_vars:
        gfs_speed = (gfs_ds.u ** 2 + gfs_ds.v ** 2) ** 0.5
    else:
        print("Warning: GFS missing v component, using only u component")
        gfs_speed = abs(gfs_ds.u)
    gfs_speed_knots = gfs_speed * 1.94384  # Convert m/s to knots
    gfs_speed_knots.attrs['units'] = 'knots'
    gfs_speed_knots.attrs['long_name'] = 'Wind Speed'

    # Apply ocean mask to ECMWF wind speed
    if ecmwf_land_ds and 'lsm' in ecmwf_land_ds.data_vars:
        lsm = ecmwf_land_ds.lsm
        ocean_mask = lsm < 0.5
        ecmwf_speed_masked = ecmwf_speed_knots.where(ocean_mask)
        print("Applied ocean mask to ECMWF wind speed")
    else:
        ecmwf_speed_masked = ecmwf_speed_knots
        print("No ocean mask applied - LSM not available")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ecmwf_speed_masked.isel(step=ecmwf_step).plot(ax=ax1, cmap='viridis')
    ax1.set_title(f'ECMWF Wind Speed (Ocean Only)\n{ecmwf_valid_times[ecmwf_step]}')

    gfs_speed_knots.isel(step=gfs_step).plot(ax=ax2, cmap='viridis')
    ax2.set_title(f'GFS Wind Speed\n{gfs_valid_times[gfs_step]}')
    plt.tight_layout()
    plt.show()

    print("\nComparison plot completed with ocean masking! (Wind speeds in knots)")


if __name__ == '__main__':
    ecmf_gfs_wind_speed_comparison()