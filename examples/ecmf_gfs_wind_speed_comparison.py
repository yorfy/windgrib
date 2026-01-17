"""comparison of wind speed forecast from ECMWF and GFS"""
from matplotlib import pyplot as plt
from windgrib import Grib


def wind_speed(name, u, v):
    """calculate wind speed in knots from u and v components"""
    speed = 1.94384 *  (u ** 2 + v ** 2) ** 0.5
    speed.attrs['units'] = 'knots'
    speed.attrs['long_name'] = f'{name} Wind Speed'
    return speed


if __name__ == '__main__':
    print("\n=== WindGrib Example: GFS/ECMWF Wind Speed Comparison ===")

    print("\n=== GFS ===")
    gfs_gb = Grib()
    # Get wind subset only for current step
    gfs_gbs = gfs_gb['wind'][gfs_gb.current_step]
    gfs_gbs.download()
    gfs_wind = gfs_gbs.ds
    # Convert longitude from (0, 360) to (-180, 180)
    gfs_wind = gfs_wind.assign_coords(longitude=((gfs_wind.longitude + 180) % 360) - 180)
    gfs_wind = gfs_wind.sortby('longitude')

    print("\n=== ECMWF ===")
    ecmwf_gb = Grib(model='ecmwf_ifs')
    # Get wind subset only for current step
    ecmwf_gbs = ecmwf_gb['wind'][ecmwf_gb.current_step]
    ecmwf_gbs.download()
    ecmwf_wind = ecmwf_gbs.ds
    # Apply ocean mask
    ecmwf_gb['land'].download()
    ecmwf_land = ecmwf_gb['land'].ds
    ocean_mask = ecmwf_land.lsm < 0.5
    ecmwf_wind = ecmwf_wind.where(ocean_mask.values)

    print("\n=== Wind Speed Comparison ===")
    # Calculate wind speed and convert m/s to knots
    gfs_wind_speed = wind_speed('GFS', gfs_wind.u, gfs_wind.v)
    ecmwf_wind_speed = wind_speed('ECMWF', ecmwf_wind.u10, ecmwf_wind.v10)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    gfs_wind_speed.plot(ax=ax1, cmap='viridis', vmin=0, vmax=40)
    ecmwf_wind_speed.plot(ax=ax2, cmap='viridis', vmin=0, vmax=40)
    fig.suptitle(f"GFS and ECMWF wind speed comparison at {gfs_gb.timestamp}")
    plt.tight_layout()
    plt.savefig('../docs/images/wind_speed_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Wind speed comparison plot saved to wind_speed_comparison.png")
    plt.show()
