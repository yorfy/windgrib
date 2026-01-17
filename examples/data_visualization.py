"""Example of data visualization with matplotlib using xarray"""
from windgrib import Grib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load data
    print("Loading data...")
    gb = Grib(model='gfswave')
    gb.download()
    wind_data = gb['wind'].ds

    # Calculate wind speed
    print("Calculating wind speed...")
    wind_speed = 1.94384 * np.sqrt(wind_data.u ** 2 + wind_data.v ** 2)
    wind_speed.attrs['units'] = 'knots'

    # Plot Wind speed - First time step
    wind_speed.isel(step=0).plot(cmap='viridis')
    plt.tight_layout()

    # Save image
    plt.savefig('../docs/images/wind_visualization.png', dpi=300, bbox_inches='tight')
