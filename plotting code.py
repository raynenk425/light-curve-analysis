import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt

# Load data and filter using your structured array column names
STAR_DATA = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
STAR_DATA = STAR_DATA[(STAR_DATA['flags'] == 0) & (STAR_DATA['Combined_outliers'] == 0)]

x = STAR_DATA['date']
y = STAR_DATA['calibrated_magnitude']
filters = STAR_DATA['filter']

# Your original loop-based colour assignment logic
colours = []
for f in filters:
    if f == 'V':
        colours.append('green')
    elif f == 'R':
        colours.append('crimson')
    elif f == 'I':
        colours.append('black')
    else:
        colours.append('grey')

plt.figure(figsize=(10, 6))

plt.scatter(x, y, color=colours, alpha=0.7, edgecolors='black')

plt.xlabel('DATE')
plt.ylabel('Calibrated Magnitude')
plt.title('Light Curve: 3216432563376802688 (Original Logic Adapted)')

plt.gca().invert_yaxis()

plt.grid(True, linestyle='--', alpha=0.5)

plt.show()