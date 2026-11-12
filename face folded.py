import matplotlib

matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

# 1. Load and filter the data
raw_data = np.load(
    '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
clean_data = raw_data[(raw_data['flags'] == 0) & (raw_data['Combined_outliers'] == 0)]

# 2. Find the ONE Master Period using the V-band
mask_v = (clean_data['filter'] == 'V')
t_v = clean_data['date'][mask_v]
y_v = clean_data['calibrated_magnitude'][mask_v]
dy_v = clean_data['calibrated_error'][mask_v]

# Use Generalized Lomb-Scargle (weighted by errors) to find the true period
ls = LombScargle(t_v, y_v, dy_v)
frequency, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=5.0)
master_period = 1.0 / frequency[np.argmax(power)]
print(f"Master Rotation Period (from V-band): {master_period:.4f} days")

# 3. Plotting Setup
filter_styles = {
    'V': {'color': 'green', 'label': 'V-band'},
    'R': {'color': 'crimson', 'label': 'R-band'},
    'I': {'color': 'black', 'label': 'I-band'}
}

plt.figure(figsize=(10, 6))

# 4. Loop through, fit the sine wave, and plot
for f_band in ['V', 'R', 'I']:
    mask = (clean_data['filter'] == f_band)
    if not np.any(mask):
        continue

    t = clean_data['date'][mask]
    y = clean_data['calibrated_magnitude'][mask]
    dy = clean_data['calibrated_error'][mask]


    # --- UPGRADED: SINE WAVE FITTING ---
    def sine_model(time_array, mean, amplitude, phase_shift):
        return mean + amplitude * np.sin((2.0 * np.pi * time_array / master_period) + phase_shift)


    guess_mean = np.median(y)
    guess_amp = (np.max(y) - np.min(y)) / 2.0

    best_rms = np.inf
    best_params = None

    # Test 10 different starting phases to prevent getting stuck in a local minimum
    for phase_guess in np.linspace(0, 2 * np.pi, 10):
        initial_guesses = [guess_mean, guess_amp, phase_guess]
        try:
            # Bounds force the amplitude to remain positive and keep the mean realistic
            params, _ = curve_fit(
                sine_model, t, y, sigma=dy, p0=initial_guesses,
                bounds=([guess_mean - 1.5, 0.0, -np.inf], [guess_mean + 1.5, guess_amp * 3.0, np.inf])
            )

            # Calculate how well this specific phase shift fits the data (RMS error)
            residuals = y - sine_model(t, *params)
            rms = np.sqrt(np.mean(residuals ** 2))

            # Keep the parameters that produce the lowest error
            if rms < best_rms:
                best_rms = rms
                best_params = params
        except RuntimeError:
            continue

    if best_params is None:
        print(f"Warning: Sine fit failed entirely for {f_band}-band. Using raw guesses.")
        best_params = [guess_mean, guess_amp, 0.0]
    # -----------------------------------

    # Calculate the phase for the raw scatter points
    phase = (t / master_period) % 1.0
    style = filter_styles[f_band]

    # Plot the raw scatter data twice (0 to 1, and 1 to 2)
    plt.scatter(phase, y, color=style['color'], alpha=0.4, edgecolors='black', s=20, label=f"{style['label']} Data")
    plt.scatter(phase + 1.0, y, color=style['color'], alpha=0.4, edgecolors='black', s=20)

    # Plot the smooth sine curve
    # Create a dummy timeline that spans exactly 2 rotations to draw a perfectly smooth line
    t_smooth = np.linspace(0, 2 * master_period, 200)
    y_smooth = sine_model(t_smooth, *best_params)
    phase_smooth = t_smooth / master_period  # This perfectly maps 0 to 2

    plt.plot(phase_smooth, y_smooth, color=style['color'], linewidth=2.5, linestyle='--', label=f"{style['label']} Fit")

# Formatting
plt.xlabel('Rotational Phase')
plt.ylabel('Calibrated Magnitude')
plt.title(f'Multi-Band Phase-Folded Light Curve with Sine Fits\nMaster Period = {master_period:.4f} days')

plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)

# Place legend outside the plot so it doesn't cover the data
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
