import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

np.random.seed(42)

# 1. Load and clean V-band data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy'
)
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]
v_data = clean_data[clean_data['filter'] == 'V']

# 2. Match the 41.4-day window
start_date = 2460626.2
end_date = 2460667.6
slice_data = v_data[(v_data['date'] >= start_date) & (v_data['date'] <= end_date)]
t = slice_data['date']
y_real = slice_data['calibrated_magnitude']
dy = slice_data['calibrated_error']

# 3. Fit weighted sine wave directly to the real slice at the injected period
injected_period = 1.9262
ls_real = LombScargle(t, y_real, dy)

y_perf = ls_real.model(t, 1.0 / injected_period)
t_smooth = np.linspace(start_date, end_date, 2000)
y_smooth = ls_real.model(t_smooth, 1.0 / injected_period)
amplitude = (np.max(y_smooth) - np.min(y_smooth)) / 2.0

# 4. Add Gaussian photometric noise (mean = y_perf, std = dy)
y_noisy = np.random.normal(loc=y_perf, scale=dy)

# 5. High-resolution Lomb-Scargle (sorted by period ascending)
frequency = np.linspace(1.0 / 3.0, 5.0, 60000)
period = 1.0 / frequency
sort_idx = np.argsort(period)
period = period[sort_idx]

power_noisy = LombScargle(t, y_noisy, dy).power(frequency)[sort_idx]
power_real = ls_real.power(frequency)[sort_idx]

peak_indices, _ = find_peaks(power_noisy, height=0.08, prominence=0.03)
sorted_indices = np.argsort(power_noisy[peak_indices])[::-1][:5]
top_periods = period[peak_indices][sorted_indices]
top_powers = power_noisy[peak_indices][sorted_indices]

print(f"\n--- Synthetic V-Band Slice ({start_date} to {end_date} | N={len(t)} | Amp={amplitude:.3f}) ---")
for i, (p, pwr) in enumerate(zip(top_periods, top_powers)):
    print(f"Peak {i + 1}: Period = {p:.4f} days | Power = {pwr:.4f}")

# 6. Side-by-side comparison plot
fig, (ax_lc, ax_ls) = plt.subplots(1, 2, figsize=(15, 6))

ax_lc.plot(t_smooth, y_smooth, color='blue', lw=1.2, alpha=0.6,
           label=f'Injected Sine (P={injected_period}d, A={amplitude:.3f})')
ax_lc.errorbar(t, y_noisy, yerr=dy, fmt='none', ecolor='grey', elinewidth=0.9,
               capsize=2, alpha=0.5, label='±1σ Error (dy)')
ax_lc.scatter(t, y_real, color='red', alpha=0.5, s=15, label='Real Data')
ax_lc.scatter(t, y_noisy, color='green', edgecolor='black', lw=0.4, alpha=0.8,
              s=20, zorder=4, label='Noisy Synthetic Points')

ax_lc.set_xlim(start_date, end_date)
ax_lc.invert_yaxis()
ax_lc.set_xlabel('Julian Date')
ax_lc.set_ylabel('Magnitude')
ax_lc.set_title(f'Synthetic V-Band Light Curve ({end_date - start_date:.1f} Days | N={len(t)})')
ax_lc.grid(True, linestyle='--', alpha=0.5)
ax_lc.legend(loc='upper right', fontsize=8)

ax_ls.plot(period, power_real, color='black', linestyle='--', lw=1.0, alpha=0.45,
           label='Real Slice Periodogram')
ax_ls.plot(period, power_noisy, color='green', lw=1.2, label='Noisy Synthetic Periodogram')
if len(top_periods) > 0:
    ax_ls.plot(top_periods, top_powers, 'ro', markersize=5, label='Synthetic Local Maxima')
    top_peak_str = f"{top_periods[0]:.4f} days"
else:
    top_peak_str = "None"

ax_ls.axvline(x=injected_period, color='blue', linestyle=':', alpha=0.6,
              label=f'Injected ({injected_period}d)')
ax_ls.axvline(x=2/3, color='red', linestyle=':', alpha=0.5, label='16h Alias (~0.667d)')
ax_ls.axvline(x=1.0, color='orange', linestyle=':', alpha=0.6, label='1.0d Line')

ax_ls.set_xlim(0, 3)
ax_ls.set_ylim(bottom=0)
ax_ls.set_xlabel('Period (days)')
ax_ls.set_ylabel('Lomb-Scargle Power')
ax_ls.set_title(f'Synthetic vs Real V-Band Periodogram (N={len(t)})\nTop Synthetic Peak: {top_peak_str}')
ax_ls.grid(True, linestyle='--', alpha=0.5)
ax_ls.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
