import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# 1. Load and clean the master data
raw_data = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

# 2. Extract timestamps, magnitudes, and errors for the full V-band
mask_v = (clean_data['filter'] == 'V')
t = clean_data['date'][mask_v]
y_real = clean_data['calibrated_magnitude'][mask_v]
dy = clean_data['calibrated_error'][mask_v]

# 3. Define the Synthetic Injection Parameters
injected_period = 1.9250
mean_mag = np.mean(y_real) # The baseline magnitude of the star (~14.3)
amplitude = np.std(y_real) # The realistic physical variation of the star
num_simulations = 100

print(f"Injecting a {injected_period}-day sine wave with REALISTIC NOISE into {len(t)} real timestamps...")
print(f"Running {num_simulations} simulations on a HIGH-RESOLUTION grid. This will take a moment...")

# 4. Create the HIGH-RESOLUTION frequency grid (Dr. Froebrich's Fix)
frequency = np.linspace(1/3, 5.0, 60000)
summed_power = np.zeros_like(frequency)

# 5. The Monte Carlo Loop
for i in range(num_simulations):
    # Randomize the starting phase of the rotation
    random_phase = np.random.uniform(0, 2 * np.pi)

    # Generate the perfect sine wave centered on the real star's magnitude
    y_sim_perfect = mean_mag + amplitude * np.sin((2 * np.pi * t / injected_period) + random_phase)

    # ====================================================================
    # DR. FROEBRICH'S EXACT NOISE INJECTION:
    # loc (mean) = the exact sine wave value at that timestamp
    # scale (standard deviation) = the exact photometric error (dy)
    # ====================================================================
    y_sim_noisy = np.random.normal(loc=y_sim_perfect, scale=dy)

    # Calculate the periodogram for this simulation using the custom grid
    # Astropy's LombScargle automatically handles the mean_mag offset mathematically
    ls = LombScargle(t, y_sim_noisy, dy)
    power = ls.power(frequency)

    # Add it to the running total
    summed_power += power

# 6. Average the results
average_power = summed_power / num_simulations

# 7. Plot the Realistic Synthetic Periodogram
plt.figure(figsize=(12, 6))
plt.plot(1.0 / frequency, average_power, color='purple', linewidth=1.5)

plt.xlim(0, 3)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Period (days)')
plt.ylabel('Average Lomb-Scargle Power')
plt.title(f'Realistic Synthetic Injection (Averaged {num_simulations}x)\nInjected: {injected_period}d | Gaussian Noise (μ = Sine, σ = dy)')

# Mark the exact locations of the features we are investigating
plt.axvline(x=injected_period, color='green', linestyle='--', alpha=0.6, label="Injected True Period (~1.925d)")
plt.axvline(x=0.66, color='red', linestyle='--', alpha=0.6, label="Expected 16h Alias (~0.66d)")
plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label="1.0d Sidereal Line")

plt.legend()
plt.tight_layout()
plt.show()
