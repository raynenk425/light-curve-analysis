import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# Load and filter out bad photometry
raw_data = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[raw_data['flags'] == 0]

# Isolate a single band for the periodogram (e.g., V-band), BY CHANGING THE mask_v value, to either V, R, I, we can see the lomb scargle for each filter
mask_v = (clean_data['filter'] == 'V')

# Extract time, magnitude, and uncertainty for the V-band
t = clean_data['date'][mask_v]
y = clean_data['calibrated_magnitude'][mask_v]
dy = clean_data['calibrated_error'][mask_v] # Incorporating the measurement error improves the fit

# Calculate the Lomb-Scargle periodogram
# autopower() automatically determines the best frequency grid based on your time sampling
ls = LombScargle(t, y, dy)
frequency, power = ls.autopower(minimum_frequency=0.01, maximum_frequency=5.0)

# Identify the strongest periodic signal
best_frequency = frequency[np.argmax(power)]
best_period = 1.0 / best_frequency
print(f"Dominant Period (V-band): {best_period:.4f} days")

# Plot the Power Spectrum (Periodogram)
plt.figure(figsize=(10, 6))

# Plotting against Period (1/frequency) is usually more intuitive for stellar rotation
plt.plot(1.0 / frequency, power, color='green', linewidth=1.5)

plt.xlabel('Period (days)')
plt.ylabel('Lomb-Scargle Power')
plt.title(f'V-Band Periodogram\nPeak Period: {best_period:.4f} days')

# Focus the x-axis on a reasonable stellar rotation range (e.g., 0 to 25 days)
plt.xlim(0, 25)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()