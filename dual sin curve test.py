import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# 1. Load and clean the master data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

mask_v = (clean_data['filter'] == 'V')
t = clean_data['date'][mask_v]
y_real = clean_data['calibrated_magnitude'][mask_v]
dy = clean_data['calibrated_error'][mask_v]

# Base physical parameters
mean_mag = np.mean(y_real)
base_amp = np.std(y_real)

# ==============================================================================
# EXPERIMENT CONTROLS
# ==============================================================================
# Period 1 (The primary ~1.925 peak)
period_1 = 1.9250
amplitude_1 = base_amp * 1.0  # Currently at 100% strength

# Period 2 (The secondary peak you found that is > 2 days)
# REPLACE 2.0700 WITH YOUR EXACT SECOND PEAK!
period_2 = 2.0695
amplitude_2 = base_amp * 0.5  # Currently at 50% strength of the main peak

# TO TEST PERIOD 2 ALONE: Set amplitude_1 = 0.0
# TO TEST BOTH (TWO-SPOT TEST): Leave both amplitudes > 0
# ==============================================================================

num_simulations = 50  # Lowered to 50 for faster testing while you tweak amplitudes

print(f"Injecting P1 = {period_1}d and P2 = {period_2}d into real timestamps...")
print("Running High-Resolution Monte Carlo... please wait.")

# High-resolution frequency grid
frequency = np.linspace(1 / 3, 5.0, 100000)
summed_power = np.zeros_like(frequency)

for i in range(num_simulations):
    # Randomize the locations of both spots independently
    phase_1 = np.random.uniform(0, 2 * np.pi)
    phase_2 = np.random.uniform(0, 2 * np.pi)

    # Generate the two distinct waves
    wave_1 = amplitude_1 * np.sin((2 * np.pi * t / period_1) + phase_1)
    wave_2 = amplitude_2 * np.sin((2 * np.pi * t / period_2) + phase_2)

    # Combine the waves and center them on the star's actual magnitude
    y_sim_perfect = mean_mag + wave_1 + wave_2

    # Inject realistic telescope noise
    y_sim_noisy = np.random.normal(loc=y_sim_perfect, scale=dy)

    ls = LombScargle(t, y_sim_noisy, dy)
    power = ls.power(frequency)
    summed_power += power

average_power = summed_power / num_simulations

# Plotting the Results
plt.figure(figsize=(12, 6))
plt.plot(1.0 / frequency, average_power, color='purple', linewidth=1.5)

plt.xlim(0, 3)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Period (days)')
plt.ylabel('Average Lomb-Scargle Power')
plt.title(
    f'Dual Sine Wave Injection (Averaged {num_simulations}x)\nP1: {period_1}d (Amp: {amplitude_1 / base_amp:.1f}x) | P2: {period_2}d (Amp: {amplitude_2 / base_amp:.1f}x)')

plt.axvline(x=period_1, color='green', linestyle='--', alpha=0.7, label=f"Injected P1 ({period_1}d)")
plt.axvline(x=period_2, color='blue', linestyle='--', alpha=0.7, label=f"Injected P2 ({period_2}d)")
plt.axvline(x=0.66, color='red', linestyle='--', alpha=0.5, label="Expected 16h Alias")

plt.legend()
plt.tight_layout()
plt.show()
