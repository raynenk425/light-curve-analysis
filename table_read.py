import numpy as np
import pandas as pd

file_path = '/Users/raynenkoonan/Desktop/Universtity/YEAR 4/STAR ANALYSIS MASTERS PROJECT/lc_3216432563376802688.npy'
STAR_DATA = np.load(file_path)

# Convert the numpy array into a readable table (DataFrame)
df = pd.DataFrame(STAR_DATA)

# Print the first 10 rows as a formatted table
print(df.head(10))

# Summary statistics (min, max, mean for each column)
print(df.describe())

# OPTIONAL: Save as a CSV file on your Desktop so you can open it in Excel/Numbers
df.to_csv('/Users/raynenkoonan/Desktop/star_data_readable.csv', index=False)
