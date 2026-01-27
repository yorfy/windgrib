""" Download wind subset for the next 10 hours from current step """
from windgrib import Grib

if __name__ == '__main__':

    print("\n====Downloading GFS wind subset for limited steps====")
    gb = Grib()
    print(f"Available steps: {gb.step}")
    # GFS wave model delivers one step per hour until step 120 then 1 step every 3 hours until step 384

    # Select wind subset for the next 10 hours from current step
    # (combination of boolean indexing and slice since there is 1 step per hour)
    wind_subset = gb['wind'][gb.step >= gb.current_step][0:10]
    # Download only this subset
    wind_subset.download()

    # Access the downloaded wind data
    wind_data = wind_subset.ds
    print(f"Downloaded steps: {wind_data.step.values}")
    print(f"Variables: {wind_data.data_vars}")

    print("\n====Downloading ECMWF wind subset for limited steps====")
    gb = Grib(model='ecmwf_ifs')
    print(f"Available steps: {gb.step}")
    # ECMWF ifs model delivers one step every 3 hours until step 144 then 1 step every 6 hours until step 360

    # Select wind subset for the next 10 hours from current step
    # (using only boolean indexing because there are missing steps)
    valid_steps = (gb.step >= gb.current_step) & (gb.step < gb.current_step + 10)
    wind_subset = gb['wind'][valid_steps]

    # Download only this subset
    wind_subset.download()

    # Access the downloaded wind data
    wind_data = wind_subset.ds
    print(f"Downloaded steps: {wind_data.step.values}")
    print(f"Variables: {wind_data.data_vars}")
