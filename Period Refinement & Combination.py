import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# 1. Load and clean the master data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

t_master, y_master, dy_master = [], [], []

print("--- Normalizing and Merging Filters ---")
# 2. Normalize V, R, and I bands so they share the same baseline and scale
for band in ['V', 'R', 'I']:
    mask = (clean_data['filter'] == band)
    t_band = clean_data['date'][mask]
    y_band = clean_data['calibrated_magnitude'][mask]
    dy_band = clean_data['calibrated_error'][mask]

    # Zero-center (subtract mean) and scale amplitude (divide by standard deviation)
    mean_mag = np.mean(y_band)
    amplitude = np.std(y_band - mean_mag)

    y_normalized = (y_band - mean_mag) / amplitude
    dy_normalized = dy_band / amplitude  # Scale errors by the same factor

    t_master.extend(t_band)
    y_master.extend(y_normalized)
    dy_master.extend(dy_normalized)

    print(f"{band}-Band: {len(t_band)} points merged.")

t_master = np.array(t_master)
y_master = np.array(y_master)
dy_master = np.array(dy_master)

print(f"Total Master Dataset: {len(t_master)} observations.\n")

# 3. Define a High-Precision, Zoomed Frequency Grid
# We know the peak is near 1.925. We zoom strictly between 1.91 and 1.94 days.
min_period = 1.91
max_period = 1.94
print(f"Executing High-Precision Zoom between {min_period} and {max_period} days...")

# Create 100,000 test frequencies in this tiny window
min_freq = 1.0 / max_period
max_freq = 1.0 / min_period
zoomed_frequencies = np.linspace(min_freq, max_freq, 100000)

# 4. Run Lomb-Scargle on the combined data using the custom grid
ls = LombScargle(t_master, y_master, dy_master)
zoomed_power = ls.power(zoomed_frequencies)

# 5. Extract the ultimate period
best_idx = np.argmax(zoomed_power)
ultimate_frequency = zoomed_frequencies[best_idx]
ultimate_period = 1.0 / ultimate_frequency

print(f"\n=========================================")
print(f"ULTIMATE MASTER PERIOD: {ultimate_period:.6f} days")
print(f"=========================================\n")

# 6. Plot the Zoomed Peak
plt.figure(figsize=(10, 6))
plt.plot(1.0 / zoomed_frequencies, zoomed_power, color='black', linewidth=2)

# Mark the exact peak
plt.axvline(ultimate_period, color='red', linestyle='--', alpha=0.7)
plt.text(ultimate_period + 0.0005, max(zoomed_power) * 0.95, f"{ultimate_period:.5f} d", color='red', fontsize=12,
         fontweight='bold')

plt.xlabel('Period (days)')
plt.ylabel('Combined Lomb-Scargle Power')
plt.title('High-Precision Multi-Band Periodogram\n(Normalized V + R + I Bands)')
plt.grid(True, linestyle='--', alpha=0.5)

# Keep the x-axis tight around the peak
plt.xlim(min_period, max_period)
plt.tight_layout()
plt.show()
