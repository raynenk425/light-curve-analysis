import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# 1. Load and clean the master data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

# 2. Extract timestamps, magnitudes, and errors for the full V-band
mask_v = (clean_data['filter'] == 'V')
t = clean_data['date'][mask_v]
y = clean_data['calibrated_magnitude'][mask_v]  # <--- ADD THIS LINE
dy = clean_data['calibrated_error'][mask_v]

# 3. Define the Synthetic Injection Parameters
injected_period = 1.9250  # The presumed TRUE rotation period
amplitude = 0.25  # Roughly matches the V-band spot depth you found earlier
num_simulations = 100  # Run 100 random phases to smooth out the noise

print(f"Injecting a {injected_period}-day sine wave into {len(t)} real timestamps...")
print(f"Running {num_simulations} simulations. This may take 10-20 seconds...")

# 4. Create a consistent frequency grid using the REAL data
ls_dummy = LombScargle(t, y, dy)  # <--- CHANGED FROM np.zeros_like(t)
frequency, _ = ls_dummy.autopower(minimum_frequency=1/3, maximum_frequency=5.0)
summed_power = np.zeros_like(frequency)

# 5. The Monte Carlo Loop
for i in range(num_simulations):
    # Randomize the starting phase of the rotation
    random_phase = np.random.uniform(0, 2 * np.pi)

    # Generate the perfect sine wave for these exact timestamps
    y_sim = amplitude * np.sin((2 * np.pi * t / injected_period) + random_phase)

    # Add Gaussian noise based on the actual telescope measurement errors
    y_sim_noisy = y_sim + np.random.normal(0, dy)

    # Calculate the periodogram for this simulation
    ls = LombScargle(t, y_sim_noisy, dy)
    power = ls.power(frequency)

    # Add it to the running total
    summed_power += power

# 6. Average the results
average_power = summed_power / num_simulations

# 7. Plot the Synthetic Periodogram
plt.figure(figsize=(12, 6))
plt.plot(1.0 / frequency, average_power, color='purple', linewidth=1.5)

plt.xlim(0, 3)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Period (days)')
plt.ylabel('Average Lomb-Scargle Power')
plt.title(
    f'Synthetic Sine Wave Injection Test (Averaged {num_simulations}x)\nInjected Signal: {injected_period} days | Real V-Band Timestamps')

# Optional: Add vertical lines where we expect the aliases to mathematically appear
plt.axvline(x=injected_period, color='green', linestyle='--', alpha=0.6, label="Injected True Period (~1.92d)")
plt.axvline(x=0.66, color='red', linestyle='--', alpha=0.6, label="Expected 16h Alias (~0.66d)")
plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.6, label="Expected Sidereal Alias (~1.0d)")

plt.legend()
plt.tight_layout()
plt.show()
