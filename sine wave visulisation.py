import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# 1. Load and clean the data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

# 2. Extract full V-band for consistent baseline mean and amplitude
mask_v = (clean_data['filter'] == 'V')
v_data = clean_data[mask_v]
mean_mag = np.mean(v_data['calibrated_magnitude'])
amplitude = np.std(v_data['calibrated_magnitude'])

# 3. Slice to a 30-day window
start_date = 2460626.0
end_date = start_date + 41.4
date_mask = (v_data['date'] >= start_date) & (v_data['date'] <= end_date)
slice_data = v_data[date_mask]

t = slice_data['date']
dy = slice_data['calibrated_error']

# 4. Generate synthetic sine wave + Gaussian noise (mean = sine, std = dy)
injected_period = 1.9262
phase = 0.0

y_perf = mean_mag + amplitude * np.sin((2 * np.pi * t / injected_period) + phase)
y_noisy = np.random.normal(loc=y_perf, scale=dy)

# Continuous curve for visual reference
t_smooth = np.linspace(start_date, end_date, 2000)
y_smooth = mean_mag + amplitude * np.sin((2 * np.pi * t_smooth / injected_period) + phase)

# 5. High-resolution Lomb-Scargle on the 30-day synthetic slice
frequency = np.linspace(1 / 3, 5.0, 60000)
ls = LombScargle(t, y_noisy, dy)
power = ls.power(frequency)
period = 1.0 / frequency

peak_indices, _ = find_peaks(power, height=0.1, prominence=0.05)
peak_periods = period[peak_indices]
peak_powers = power[peak_indices]

sorted_indices = np.argsort(peak_powers)[::-1]
top_periods = peak_periods[sorted_indices][:5]
top_powers = peak_powers[sorted_indices][:5]

print(f"\n--- Synthetic V-Band 30-Day Slice ({start_date:.1f} to {end_date:.1f} | N={len(t)}) ---")
for i, (p, pwr) in enumerate(zip(top_periods, top_powers)):
    print(f"Peak {i + 1}: Period = {p:.4f} days | Power = {pwr:.4f}")

# 6. Side-by-side plot
fig, (ax_lc, ax_ls) = plt.subplots(1, 2, figsize=(15, 6))

# Left Panel: 30-day synthetic light curve
ax_lc.plot(t_smooth, y_smooth, color='blue', linewidth=1.2, alpha=0.6, label=f'Injected Sine (P={injected_period}d)')
ax_lc.errorbar(t, y_perf, yerr=dy, fmt='none', ecolor='grey', elinewidth=1.0,
               capsize=2, alpha=0.6, label='±1σ Error (dy)')
ax_lc.scatter(t, slice_data['calibrated_magnitude'], color='red', alpha=0.5, s=15, label='Real Data')
ax_lc.scatter(t, y_noisy, color='green', edgecolor='black', alpha=0.75, s=20,
              zorder=4, label='Noisy Synthetic Points')

ax_lc.set_xlim(start_date, end_date)
ax_lc.invert_yaxis()
ax_lc.set_xlabel('Julian Date')
ax_lc.set_ylabel('Synthetic Magnitude')
ax_lc.set_title(f'Synthetic V-Band Light Curve (30 Days | N={len(t)})')
ax_lc.grid(True, linestyle='--', alpha=0.5)
ax_lc.legend(loc='upper right', fontsize=8)

# Right Panel: Lomb-Scargle periodogram
ax_ls.plot(period, power, color='green', linewidth=1.2)
if len(top_periods) > 0:
    ax_ls.plot(top_periods, top_powers, 'ro', label='Local Maximums')
    top_peak_str = f"{top_periods[0]:.4f} days"
else:
    top_peak_str = "None"

ax_ls.axvline(x=injected_period, color='blue', linestyle='--', alpha=0.6, label=f'Injected ({injected_period}d)')
ax_ls.axvline(x=0.66, color='red', linestyle='--', alpha=0.6, label='16h Alias (~0.66d)')
ax_ls.axvline(x=1.0, color='orange', linestyle='--', alpha=0.7, label='1.0d Line')

ax_ls.set_xlim(0, 3)
ax_ls.set_ylim(bottom=0)
ax_ls.set_xlabel('Period (days)')
ax_ls.set_ylabel('Lomb-Scargle Power')
ax_ls.set_title(f'Synthetic V-Band Periodogram (30 Days)\nTop Peak: {top_peak_str}')
ax_ls.grid(True, linestyle='--', alpha=0.5)
ax_ls.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
