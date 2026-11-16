import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# 1. Load and clean the data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

filter_colors = {'V': 'green', 'R': 'crimson', 'I': 'black'}

min_date = np.min(clean_data['date'])
max_date = np.max(clean_data['date'])

# 2. Set up the Matplotlib Figure with TWO panels (1 row, 2 columns)
fig, (ax_lc, ax_ls) = plt.subplots(1, 2, figsize=(15, 7))

# Adjust layout to make room for the widgets
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, wspace=0.2)

# 3. Create the GUI Widgets
ax_start = plt.axes([0.35, 0.1, 0.15, 0.05])
text_start = TextBox(ax_start, 'Start Date: ', initial=f"{min_date:.1f}")

ax_end = plt.axes([0.65, 0.1, 0.15, 0.05])
text_end = TextBox(ax_end, 'End Date: ', initial=f"{max_date:.1f}")

ax_radio = plt.axes([0.02, 0.45, 0.06, 0.15])
radio = RadioButtons(ax_radio, ('V', 'R', 'I'), activecolor='blue')


# 4. The Live Update Engine
def update(text):
    # Clear BOTH plots
    ax_lc.clear()
    ax_ls.clear()

    try:
        start_val = float(text_start.text)
        end_val = float(text_end.text)
    except ValueError:
        ax_ls.text(0.5, 0.5, "Invalid Date Input. Please enter numbers.",
                   ha='center', va='center', transform=ax_ls.transAxes, fontsize=14, color='red')
        fig.canvas.draw_idle()
        return

    target_filter = radio.value_selected
    plot_color = filter_colors.get(target_filter, 'blue')

    # --- LEFT PANEL: FULL LIGHT CURVE ---
    # Isolate all data for the selected filter to show the full timeline
    filter_mask = (clean_data['filter'] == target_filter)
    full_filter_data = clean_data[filter_mask]

    # Plot the full light curve
    ax_lc.scatter(full_filter_data['date'], full_filter_data['calibrated_magnitude'],
                  color=plot_color, edgecolor='black', alpha=0.6, s=15)

    # Draw a shaded vertical box to visually highlight the selected date range!
    ax_lc.axvspan(start_val, end_val, color='grey', alpha=0.3)

    ax_lc.set_xlabel('Julian Date')
    ax_lc.set_ylabel('Calibrated Magnitude')
    ax_lc.set_title(f"Full {target_filter}-Band Light Curve\n(Highlighted area feeds the Periodogram)")
    ax_lc.invert_yaxis()  # Magnitudes go down in astronomy
    ax_lc.grid(True, linestyle='--', alpha=0.5)

    # --- RIGHT PANEL: LOMB-SCARGLE PERIODOGRAM ---
    # Slice the data down to JUST the dates inside the highlighted box
    date_mask = (full_filter_data['date'] >= start_val) & (full_filter_data['date'] <= end_val)
    slice_data = full_filter_data[date_mask]

    t = slice_data['date']
    y = slice_data['calibrated_magnitude']
    dy = slice_data['calibrated_error']

    # Failsafe for empty selections
    if len(t) < 10:
        ax_ls.text(0.5, 0.5, f"Not enough data points (N={len(t)})\nin this date range.",
                   ha='center', va='center', transform=ax_ls.transAxes, fontsize=14, color='red')
        ax_ls.set_title(f"{target_filter}-Band Periodogram")
        fig.canvas.draw_idle()
        return

    # Run the Math
    ls = LombScargle(t, y, dy)
    # Force it to create exactly 100,000 test frequencies
    frequency = np.linspace(1 / 3, 5.0, 60000)

    # Manually calculate the power for those 100,000 specific points
    power = ls.power(frequency)
    period = 1.0 / frequency

    # Peak Finding
    peak_indices, _ = find_peaks(power, height=0.1, prominence=0.05)
    peak_periods = period[peak_indices]
    peak_powers = power[peak_indices]

    sorted_indices = np.argsort(peak_powers)[::-1]
    top_periods = peak_periods[sorted_indices][:5]
    top_powers = peak_powers[sorted_indices][:5]

    # Terminal Output
    print(f"\n--- {target_filter}-Band Peaks ({start_val} to {end_val}) ---")
    for i, (p, pwr) in enumerate(zip(top_periods, top_powers)):
        print(f"Peak {i + 1}: Period = {p:.4f} days | Power = {pwr:.4f}")

    # Plot the Periodogram
    ax_ls.plot(period, power, color=plot_color, linewidth=1.5)
    ax_ls.plot(top_periods, top_powers, "ro", label="Local Maximums")

    ax_ls.set_xlim(0, 3)
    ax_ls.set_ylim(bottom=0)
    ax_ls.grid(True, linestyle='--', alpha=0.5)
    ax_ls.set_xlabel('Period (days)')
    ax_ls.set_ylabel('Lomb-Scargle Power')
    ax_ls.legend(loc='upper right')
    ax_ls.set_title(
        f"{target_filter}-Band Periodogram | N = {len(t)} points\nDates: {start_val:.1f} to {end_val:.1f} | Top Peak: {top_periods[0]:.4f} days")

    fig.canvas.draw_idle()


# 5. Connect widgets
text_start.on_submit(update)
text_end.on_submit(update)
radio.on_clicked(update)

update(None)

plt.show()
