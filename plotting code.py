import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt

STAR_DATA = np.load('/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy')
STAR_DATA = STAR_DATA[STAR_DATA['flags'] == 0]
print(STAR_DATA.dtype.names)
x= STAR_DATA['date']
y= STAR_DATA['calibrated_magnitude']
filters = STAR_DATA['filter']
flags = STAR_DATA['flags']

colours = []
for f, fl in zip(filters,flags):
    if fl != 0:
        colours.append('hotpink')
    elif f == 'V':
        colours.append('green')
    elif f == 'R':
        colours.append('crimson')
    elif f == 'I':
        colours.append('black')
    else:
        colours.append('grey')




plt.figure(figsize=(8, 6))
plt.scatter(x, y, color=colours, alpha=0.7, edgecolors='black', )

plt.xlabel('DATE')
plt.ylabel('Calibrated Magnitude')
plt.title(f'DATE against CALIBRATED MAGNITUDE')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()