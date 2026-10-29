import numpy as np
import matplotlib.pyplot as plt

STAR_DATA = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')

print(STAR_DATA.dtype.names)
x= STAR_DATA['date']
y= STAR_DATA['calibrated_magnitude']


plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='crimson', alpha=0.7, edgecolors='black')

plt.xlabel('DATE')
plt.ylabel('Calibrated Magnitude')
plt.title(f'DATE against CALIBRATED MAGNITUDE')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()