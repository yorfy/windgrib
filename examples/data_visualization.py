"""Example of data visualization with matplotlib using xarray"""
from windgrib import Grib
import numpy as np
import matplotlib.pyplot as plt

    
# Load data
print("Loading data...")
grib = Grib(model='gfswave')
grib.download()
wind_data = grib['wind']

# Calculate wind speed
print("Calculating wind speed...")
wind_speed = np.sqrt(wind_data.u**2 + wind_data.v**2)

# Plot Wind speed - First time step
wind_speed.isel(step=0).plot(cmap='viridis')
plt.tight_layout()

# Save image
plt.savefig('wind_visualization.png', dpi=300, bbox_inches='tight')


