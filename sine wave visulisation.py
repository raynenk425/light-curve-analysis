import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and clean the master data
raw_data = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

# Extract timestamps, magnitudes, and errors for the V-band
mask_v = (clean_data['filter'] == 'V')
t = clean_data['date'][mask_v]
y_real = clean_data['calibrated_magnitude'][mask_v]
dy = clean_data['calibrated_error'][mask_v]

# 2. Parameters for the Synthetic Data
injected_period = 1.9250
mean_mag = np.mean(y_real)         # Center it on the actual star's magnitude (e.g., ~14.3)
amplitude = np.std(y_real)         # Match the real physical variation
fixed_phase = 0.0

# 3. Generate the Realistic Synthetic Data
# Pure sine wave shifted to the real mean magnitude
y_sim_perfect = mean_mag + amplitude * np.sin((2 * np.pi * t / injected_period) + fixed_phase)

# Add the Gaussian photometric noise based on the telescope's dy
y_sim_noisy = np.random.normal(loc=y_sim_perfect, scale=dy)

# 4. Plot the Side-by-Side Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# --- LEFT PANEL: REAL DATA ---
ax1.scatter(t, y_real, color='green', edgecolor='black', alpha=0.6, s=15)
ax1.set_title('REAL V-Band Light Curve')
ax1.set_xlabel('Julian Date')
ax1.set_ylabel('Calibrated Magnitude')
ax1.invert_yaxis() # Magnitudes go down!
ax1.grid(True, linestyle='--', alpha=0.5)

# --- RIGHT PANEL: SYNTHETIC DATA ---
ax2.scatter(t, y_sim_noisy, color='purple', edgecolor='black', alpha=0.6, s=15)
ax2.set_title(f'SYNTHETIC V-Band Light Curve\n(Injected: {injected_period}d + Real dy Noise)')
ax2.set_xlabel('Julian Date')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
