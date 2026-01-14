"""Minimal example of downloading and accessing data,
using cache and saving data to netcdf format to speedup further data reading"""

from windgrib import Grib

if __name__ == '__main__':
    # Create a GRIB instance for the GFS Wave model
    print("\n====Initiating Grib instance and looking for forecast data====")
    g = Grib(model='ecmwf_ifs')
    # g = Grib()

    # Download the data
    print("\n====Downloading GFS Wave data...====")
    g.download(clear_cache=True)
    # clear_cache=False is the default option
    # But you can use clear_cache=True to force downloading ignoring cache files

    # save to grib file for further analysis of downloaded data
    g.to_grib_file()

    # Access wind data
    wind_data = g['wind']
    print(f"====Loaded dataset====\n{wind_data}")

    # save to netcdf file to speedup further data reading
    g.to_nc()
