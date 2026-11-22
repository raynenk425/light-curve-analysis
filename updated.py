import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# 1. Load and clean the data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

filter_colors = {'V': 'green', 'R': 'crimson', 'I': 'black'}

# 2. Set up the figure
fig, (ax_lc, ax_ls) = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(bottom=0.12, left=0.12, right=0.95, wspace=0.2)

ax_radio = plt.axes([0.02, 0.45, 0.06, 0.15])
radio = RadioButtons(ax_radio, ('V', 'R', 'I'), activecolor='blue')


def update_periodogram(start_val, end_val):
    ax_ls.clear()

    target_filter = radio.value_selected
    plot_color = filter_colors.get(target_filter, 'blue')

    filter_mask = (clean_data['filter'] == target_filter)
    full_filter_data = clean_data[filter_mask]

    date_mask = (full_filter_data['date'] >= start_val) & (full_filter_data['date'] <= end_val)
    slice_data = full_filter_data[date_mask]

    t = slice_data['date']
    y = slice_data['calibrated_magnitude']
    dy = slice_data['calibrated_error']

    if len(t) < 10:
        ax_ls.text(0.5, 0.5, f"Not enough data points (N={len(t)})\nin this date range.",
                   ha='center', va='center', transform=ax_ls.transAxes, fontsize=14, color='red')
        ax_ls.set_title(f"{target_filter}-Band Periodogram")
        fig.canvas.draw_idle()
        return

    ls = LombScargle(t, y, dy)
    frequency = np.linspace(1 / 3, 5.0, 60000)
    power = ls.power(frequency)
    period = 1.0 / frequency

    peak_indices, _ = find_peaks(power, height=0.1, prominence=0.05)
    peak_periods = period[peak_indices]
    peak_powers = power[peak_indices]

    sorted_indices = np.argsort(peak_powers)[::-1]
    top_periods = peak_periods[sorted_indices][:5]
    top_powers = peak_powers[sorted_indices][:5]

    print(f"\n--- {target_filter}-Band Peaks ({start_val:.1f} to {end_val:.1f} | N={len(t)}) ---")
    for i, (p, pwr) in enumerate(zip(top_periods, top_powers)):
        print(f"Peak {i + 1}: Period = {p:.4f} days | Power = {pwr:.4f}")

    ax_ls.plot(period, power, color=plot_color, linewidth=1.5)
    if len(top_periods) > 0:
        ax_ls.plot(top_periods, top_powers, "ro", label="Local Maximums")
        top_peak_str = f"{top_periods[0]:.4f} days"
    else:
        top_peak_str = "None"

    ax_ls.set_xlim(0, 3)
    ax_ls.set_ylim(bottom=0)
    ax_ls.grid(True, linestyle='--', alpha=0.5)
    ax_ls.set_xlabel('Period (days)')
    ax_ls.set_ylabel('Lomb-Scargle Power')
    ax_ls.legend(loc='upper right')
    ax_ls.set_title(
        f"{target_filter}-Band Periodogram | N = {len(t)} points\nDates: {start_val:.1f} to {end_val:.1f} | Top Peak: {top_peak_str}")

    fig.canvas.draw_idle()


def on_xlim_change(event_ax):
    toolbar = fig.canvas.toolbar
    if toolbar is not None and toolbar.mode == 'pan/zoom':
        return
    start_val, end_val = sorted(event_ax.get_xlim())
    update_periodogram(start_val, end_val)


def draw_lightcurve(preserve_xlim=False):
    curr_xlim = ax_lc.get_xlim() if preserve_xlim else None
    ax_lc.clear()

    target_filter = radio.value_selected
    plot_color = filter_colors.get(target_filter, 'blue')

    filter_mask = (clean_data['filter'] == target_filter)
    full_filter_data = clean_data[filter_mask]

    ax_lc.scatter(full_filter_data['date'], full_filter_data['calibrated_magnitude'],
                  color=plot_color, edgecolor='black', alpha=0.6, s=15)

    ax_lc.set_xlabel('Julian Date')
    ax_lc.set_ylabel('Calibrated Magnitude')
    ax_lc.set_title(f"{target_filter}-Band Light Curve (Drag-to-Zoom)")
    ax_lc.invert_yaxis()
    ax_lc.grid(True, linestyle='--', alpha=0.5)

    if preserve_xlim and curr_xlim is not None:
        ax_lc.set_xlim(curr_xlim)

    # Reconnect after clear(), as clear() resets axis callbacks
    ax_lc.callbacks.connect('xlim_changed', on_xlim_change)


def on_radio_change(label):
    draw_lightcurve(preserve_xlim=True)
    start_val, end_val = sorted(ax_lc.get_xlim())
    update_periodogram(start_val, end_val)


radio.on_clicked(on_radio_change)

draw_lightcurve(preserve_xlim=False)
start_init, end_init = sorted(ax_lc.get_xlim())
update_periodogram(start_init, end_init)

plt.show()
