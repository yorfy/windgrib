"""
WindGrib Usage Example: GFS Atmospheric Temperature Data

This example demonstrates how to define a custom model to download
and analyze GFS atmospheric temperature data.
"""

from matplotlib import pyplot as plt
from windgrib.grib import MODELS, Grib

# Configuration for GFS atmospheric with surface temperature subset
MODELS['gfs_atmos_temperature'] = {
    'product': '.pgrb2.0p25',  # start with . to prevent considering goessimpgrb2 product
    'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
    'key': 'gfs.{date}/{h:02d}/atmos/',
    'subsets': {
        'temperature': (['TMP'], {'layer': ['surface']})
    },
    'ext': ''
}

if __name__ == '__main__':
    print("=== WindGrib Example: GFS Atmospheric Data ===\n")

    print("1. Downloading latest available GFS atmospheric data...")
    gb = Grib(model='gfs_atmos_temperature')
    gb.download()
    print("Download completed")

    print("\n2. Analyzing temperature data...")
    ds = gb['temperature'].ds
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {list(ds.dims)}")

    # Convert Kelvin to Celsius and plot temperature variation near Toulouse
    temp_celsius = ds.t - 273.15
    temp_celsius.attrs['units'] = '°C'
    temp_celsius.interp({'latitude': 43.599998, 'longitude': 1.43333}).plot()
    plt.suptitle("Temperature Variation near Toulouse")
    plt.savefig("../docs/images/temperature_variation_near_toulouse.png")
    print("Example completed")
