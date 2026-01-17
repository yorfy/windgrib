"""
WindGrib Custom Model Example

This example demonstrates how to define and use custom models with WindGrib.
It shows how to create a custom model for GFS atmospheric temperature data.
"""

from matplotlib import pyplot as plt
from windgrib.grib import MODELS, Grib


def main():
    print("=== WindGrib Custom Model Example ===\n")

    # Configuration for GFS atmospheric with surface temperature subset
    MODELS['gfs_atmos_temperature'] = {
        'product': 'pgrb2.0p25',
        'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
        'key': 'gfs.{date}/{h:02d}/atmos/',
        'ext': '',
        'subsets': {
            'temperature': (['TMP'], {'layer':'surface'})
        }
    }

    print("1. Custom model defined: gfs_atmos_temperature")
    print("   - Product: .pgrb2.0p25")
    print("   - Variables: TMP (temperature)")
    print("   - Layer: surface")

    print("\n2. Downloading latest available GFS atmospheric data...")
    gb = Grib(model='gfs_atmos_temperature')
    gb.download()
    print("Download completed")

    print("\n3. Analyzing temperature data...")

    ds = gb['temperature'].ds
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {list(ds.dims)}")

    # Get the temperature variable (check what's actually available)
    temp_var = None
    for var in ds.data_vars:
        if 'tmp' in var.lower() or 't' == var.lower():
            temp_var = var
            break

    if temp_var is None:
        print(f"Available variables: {list(ds.data_vars)}")
        temp_var = list(ds.data_vars)[0]  # Use first available variable
        print(f"Using variable: {temp_var}")

    # Convert Kelvin to Celsius and plot temperature variation near Toulouse
    temp_celsius = ds[temp_var] - 273.15
    temp_celsius.attrs['units'] = '°C'
    temp_celsius.interp({'latitude': 43.599998, 'longitude': 1.43333}).plot()
    plt.suptitle("Temperature Variation near Toulouse")
    plt.savefig("temperature_variation_near_toulouse.png")
    print("💾 Temperature plot saved to temperature_variation_near_toulouse.png")
    print("Example completed")


if __name__ == '__main__':
    main()
