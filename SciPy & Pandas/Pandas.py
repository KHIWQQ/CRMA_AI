import pandas as pd
import matplotlib.pyplot as plt

mydataset = {
'cars': ["BMW", "Volvo", "Ford"],
'passings': [3, 7, 2]
}
myvar = pd.DataFrame(mydataset)
print(myvar)
print()

# Pandas Series
a = [1, 7, 2]
myvar = pd.Series(a)
print(myvar)
print()

a = [1, 7, 2]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar)
print()

# Key/Value Objects as Series
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)
print(myvar)
print()

calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories, index = ["day1", "day2"])
print(myvar)
print()

# Pandas DataFrames
data = {
"calories": [420, 380, 390],
"duration": [50, 40, 45]
}
myvar = pd.DataFrame(data)
print(myvar)
print()

# Pandas DataFrames
data = {
"calories": [420, 380, 390],
"duration": [50, 40, 45]
}
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df)

# # Locate Row
# #refer to the row index:
# print(df.loc[0])
#
# #use a list of indexes:
# print(df.loc[[0, 1]])
#
# #Locate Named Indexes
# print(df.loc["day2"])

# Pandas Read CSV (comma separated values)
df = pd.read_csv('data.csv')
print(df.to_string())
print(df)
print()

# Pandas Read JSON
df = pd.read_json('data.json')
print(df.to_string())
print()

# Dictionary as JSON
data = {
"Duration":{
"0":60,
"1":60,
"2":60,
"3":45,
"4":45,
"5":60
},
"Pulse":{
"0":110,
"1":117,
"2":103,
"3":109,
"4":117,
"5":102
},
"Maxpulse":{
"0":130,
"1":145,
"2":135,
"3":175,
"4":148,
"5":127
},
"Calories":{
"0":409,
"1":479,
"2":340,
"3":282,
"4":406,
"5":300
}
}
df = pd.DataFrame(data)
print(df)
print()

# Pandas - Analyzing DataFrames
df = pd.read_csv('data.csv')
print(df.head())
print()

print(df.head(10))
print()

print(df.tail())
print()

print(df.info())
print()

# Pandas - Cleaning Data
# 1. Pandas - Cleaning Empty Cells
df1 = pd.read_csv('dirtydata.csv')
new_df = df1.dropna()
print(new_df.to_string())
print()

# Remove all rows with NULL values:
df = pd.read_csv('data.csv')
df.dropna(inplace = True)
print(df.to_string())
print()

# Pandas - Replace Empty Values
df1 = pd.read_csv('dirtydata.csv')
df1.fillna(130, inplace = True)
print(df1)

# Replace Only For a Specified Columns
df2 = pd.read_csv('dirtydata.csv')
df2["Calories"].fillna(130, inplace = True)
print(df2)

# Replace Using Mean, Median, or Mode
df3 = pd.read_csv('dirtydata.csv')
x = df3["Calories"].mean() #or median, mode
df3["Calories"].fillna(x, inplace = True)
print(df3)

# Pandas - Data Correlations
df = pd.read_csv('data.csv')
print(df.corr())

# Pandas - Plotting
df = pd.read_csv('data.csv')
df.plot()
plt.show()

# Scatter Plot
df = pd.read_csv('data.csv')
df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
plt.show()

# Histogram
df = pd.read_csv('data.csv')
df["Duration"].plot(kind = 'hist')
plt.show()