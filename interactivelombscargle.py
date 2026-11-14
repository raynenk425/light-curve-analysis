import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
from astropy.timeseries import LombScargle

# 1. Load and filter out bad photometry (DONE ONCE)
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

filter_colors = {'V': 'green', 'R': 'crimson', 'I': 'black'}

# Find the absolute min and max Julian Dates for the initial text box values
min_date = np.min(clean_data['date'])
max_date = np.max(clean_data['date'])

# 2. Set up the Matplotlib Figure and Axes
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(bottom=0.25, left=0.25)  # Make room for GUI

# 3. Create the Widgets (Text Boxes and Radio Buttons)
# Start Date Text Box
ax_start = plt.axes([0.35, 0.1, 0.2, 0.05])  # [left, bottom, width, height]
text_start = TextBox(ax_start, 'Start Date: ', initial=f"{min_date:.1f}")

# End Date Text Box
ax_end = plt.axes([0.7, 0.1, 0.2, 0.05])
text_end = TextBox(ax_end, 'End Date: ', initial=f"{max_date:.1f}")

# Filter Radio Buttons
ax_radio = plt.axes([0.05, 0.4, 0.1, 0.15])
radio = RadioButtons(ax_radio, ('V', 'R', 'I'), activecolor='blue')


# 4. The Live Update Engine
def update(text):
    ax.clear()  # Clear the previous plot

    # Safely try to read the text as numbers
    try:
        start_val = float(text_start.text)
        end_val = float(text_end.text)
    except ValueError:
        # If you accidentally type a letter, it warns you instead of crashing
        ax.text(0.5, 0.5, "Invalid Date Input. Please enter numbers.",
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        fig.canvas.draw_idle()
        return

    target_filter = radio.value_selected

    # Apply the masks
    date_mask = (clean_data['date'] >= start_val) & (clean_data['date'] <= end_val)
    filter_mask = (clean_data['filter'] == target_filter)
    slice_data = clean_data[date_mask & filter_mask]

    t = slice_data['date']
    y = slice_data['calibrated_magnitude']
    dy = slice_data['calibrated_error']

    # Data Density Check
    if len(t) < 10:
        ax.text(0.5, 0.5, f"Not enough data points (N={len(t)})\nin this date range.",
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title(f"{target_filter}-Band Periodogram")
        fig.canvas.draw_idle()
        return

    # Calculate Lomb-Scargle
    ls = LombScargle(t, y, dy)
    frequency, power = ls.autopower(minimum_frequency=1 / 3, maximum_frequency=5.0)

    best_period = 1.0 / frequency[np.argmax(power)]
    plot_color = filter_colors.get(target_filter, 'blue')

    # Draw the new graph
    ax.plot(1.0 / frequency, power, color=plot_color, linewidth=1.5)

    # Formatting
    ax.set_xlim(0, 3)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Lomb-Scargle Power')

    ax.set_title(
        f"{target_filter}-Band Periodogram | N = {len(t)} points\nDates: {start_val:.1f} to {end_val:.1f} | Peak Period: {best_period:.4f} days")
    fig.canvas.draw_idle()


# 5. Connect the widgets
# .on_submit means the graph updates ONLY when you press 'Enter' inside the text box
text_start.on_submit(update)
text_end.on_submit(update)
radio.on_clicked(update)  # Radio buttons update instantly when clicked

# Force the first plot to draw
update(None)

plt.show()
