import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('mtcarsDataset.csv')
df = pd.DataFrame(data)

print(data.shape)
#
print(data.head(10))

print(df.describe())

data.plot()
plt.show()