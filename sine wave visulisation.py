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

# 2. ONE window mask, used for everything (real, pure, noisy)  # NEW: reconciliation
start_date = 2460626.2
end_date = 2460667.6
win = (v_data['date'] >= start_date) & (v_data['date'] <= end_date)
slice_data = v_data[win]
t = slice_data['date']
y_real = slice_data['calibrated_magnitude']
dy = slice_data['calibrated_error']
N = len(t)

print(f"\nWindow {start_date}–{end_date}: N = {N} | first = {t.min():.3f} | last = {t.max():.3f}")
print("(This N is the one to quote. Earlier '308' came from the drag-to-zoom axis limits, not this mask.)")

# 3. Fit weighted sine to the real slice at the injected period
injected_period = 1.9262
ls_real = LombScargle(t, y_real, dy)
y_perf = ls_real.model(t, 1.0 / injected_period)
t_smooth = np.linspace(start_date, end_date, 2000)
y_smooth = ls_real.model(t_smooth, 1.0 / injected_period)
amplitude = (np.max(y_smooth) - np.min(y_smooth)) / 2.0

# 3b. Residual scatter vs reported error  # NEW
resid_std = np.std(y_real - y_perf)
median_dy = np.median(dy)
print(f"\nResidual scatter std(y_real - y_perf) = {resid_std:.4f} mag")
print(f"Median reported dy = {median_dy:.4f} mag")
print(f"Ratio dy / residual = {median_dy / resid_std:.2f}")

# 4. Noisy synthetic (mean = sine, std = dy)
y_noisy = np.random.normal(loc=y_perf, scale=dy)

# 5. Same frequency grid for all three periodograms
n_grid = 60000
frequency = np.linspace(1.0 / 3.0, 5.0, n_grid)
period = 1.0 / frequency
sort_idx = np.argsort(period)
period = period[sort_idx]

power_real = ls_real.power(frequency)[sort_idx]
power_pure = LombScargle(t, y_perf, dy).power(frequency)[sort_idx]  # NEW: no-noise sine
power_noisy = LombScargle(t, y_noisy, dy).power(frequency)[sort_idx]

# 5b. Measured local maxima near the four features of interest  # NEW
def local_max(period, power, centre, halfwidth):
    m = (period > centre - halfwidth) & (period < centre + halfwidth)
    i = np.argmax(power[m])
    return period[m][i], power[m][i]

features = {
    'main (~1.93d)': (1.93, 0.06),
    'second (~2.08d)': (2.08, 0.05),
    '16h (~0.667d)': (0.667, 0.02),
    '~1d': (1.00, 0.03),
}
curves = {'real': power_real, 'pure': power_pure, 'noisy': power_noisy}

print(f"\n{'feature':<18}{'curve':<8}{'period(d)':>11}{'power':>9}{'P/main':>9}")
table = {}
for cname, pw in curves.items():
    table[cname] = {f: local_max(period, pw, c, hw) for f, (c, hw) in features.items()}
    main_p = table[cname]['main (~1.93d)'][1]
    for f, (p, pwr) in table[cname].items():
        print(f"{f:<18}{cname:<8}{p:>11.4f}{pwr:>9.4f}{pwr / main_p:>9.3f}")

# Ratios Dirk cares about
for cname in curves:
    r12 = table[cname]['main (~1.93d)'][1] / table[cname]['second (~2.08d)'][1]
    r16 = table[cname]['main (~1.93d)'][1] / table[cname]['16h (~0.667d)'][1]
    print(f"{cname:<6} main/second = {r12:.2f} | main/16h = {r16:.2f}")

# 6. Plot
fig, (ax_lc, ax_ls) = plt.subplots(1, 2, figsize=(15, 6.5))

ax_lc.plot(
    t_smooth, y_smooth, color='blue', lw=1.2, alpha=0.6,
    label=f'Injected Sine (P={injected_period}d, A={amplitude:.3f})'
)
ax_lc.errorbar(
    t, y_noisy, yerr=dy, fmt='none', ecolor='grey',
    elinewidth=0.9, capsize=2, alpha=0.5, label='±1σ Error (dy)'
)
ax_lc.scatter(t, y_real, color='red', alpha=0.5, s=15, label='Real Data')
ax_lc.scatter(
    t, y_noisy, color='green', edgecolor='black', lw=0.4,
    alpha=0.8, s=20, zorder=4, label='Noisy Synthetic Points'
)
ax_lc.set_xlim(start_date, end_date)
ax_lc.invert_yaxis()
ax_lc.set_xlabel('Julian Date')
ax_lc.set_ylabel('Magnitude')
ax_lc.set_title(f'Synthetic V-Band Light Curve ({end_date - start_date:.1f} Days | N={N})')
ax_lc.grid(True, linestyle='--', alpha=0.5)
ax_lc.legend(loc='upper right', fontsize=8)

ax_ls.plot(period, power_real, color='black', linestyle='--', lw=1.0, alpha=0.45, label='Real Slice')
ax_ls.plot(period, power_pure, color='purple', lw=1.0, alpha=0.7, label='Pure Sine (no noise)')  # NEW
ax_ls.plot(period, power_noisy, color='green', lw=1.2, label='Noisy Synthetic')
for f, (p, pwr) in table['noisy'].items():
    ax_ls.plot(p, pwr, 'ro', markersize=4)

ax_ls.axvline(x=injected_period, color='blue', linestyle=':', alpha=0.6, label=f'Injected ({injected_period}d)')
ax_ls.axvline(x=2 / 3, color='red', linestyle=':', alpha=0.5, label='16h Alias (~0.667d)')
ax_ls.axvline(x=1.0, color='orange', linestyle=':', alpha=0.6, label='1.0d Line')
ax_ls.set_xlim(0, 3)
ax_ls.set_ylim(bottom=0)
ax_ls.set_xlabel('Period (days)')
ax_ls.set_ylabel('Lomb-Scargle Power')
ax_ls.set_title(
    f'Real vs Pure vs Noisy Synthetic V-Band Periodogram (N={N})\n'
    f"Noisy main peak: {table['noisy']['main (~1.93d)'][0]:.4f} d"
)
ax_ls.grid(True, linestyle='--', alpha=0.5)
ax_ls.legend(loc='upper right', fontsize=8)

# 7. Caption  # NEW
caption = (
    f"V-band, JD {start_date:.1f}–{end_date:.1f} ({end_date - start_date:.1f} d), N={N}. "
    f"Injected sine P={injected_period} d, A={amplitude:.3f} mag (mean/amplitude/phase from weighted "
    f"LS fit to real slice). Noise model: each point drawn from N(mean=sine value, σ=per-point dy). "
    f"Single realisation (seed 42). Lomb-Scargle on {n_grid} uniform frequencies, 0.2–3.0 d, "
    f"astropy standard normalisation, dy used as weights for all three curves."
)
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=8, wrap=True)

plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
