import  pandas as pd
#Insert the correct Pandas method to create a Series.
mylist = [1,7,2]
myseries = pd.Series(mylist)
print(myseries)
#Insert the correct syntax to return the first value of a Pandas Series called myseries
print(myseries[0])
#Insert the correct syntax to add the labels "x", "y", and "z" to a Pandas Series.
myvar = pd.Series(mylist, index = ["x", "y", "z"])
print(myvar)
#Insert the correct Pandas method to create a DataFrame.
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data)
print(df)
#Insert the correct syntax to return the first row of a DataFrame.
print(df.loc[0])
#Insert the correct syntax for loading CSV files into a DataFrame.
df = pd.read_csv('data.csv')
print(df)
#Insert the correct syntax to return the entire DataFrame.
df = pd.read_csv('data.csv')
print(df.to_string())
#Insert the correct syntax for loading JSON files into a DataFrame.
df = pd.read_json('data.json')
print(df)
#Insert the correct syntax for loading a Python dictionary called "data" into a DataFrame.
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
#Insert the correct syntax for returning the headers and the first 10 rows of a DataFrame.
df = pd.read_csv('data.csv')
print(df.head(10))
#When using the head() method, how many rows are returned if you do not specify the number?
#The headers and 5 rows
df = pd.read_csv('data.csv')
print(df.head())
#The head() method returns the first rows, what method returns the last rows?
print(df.tail())

#Pandas Cleansing
#Insert the correct syntax for removing rows with empty cells.
df = pd.read_csv('data.csv')
new_df = df.dropna()
print(new_df.to_string())
#Insert the correct syntax for replacing empty cells with the value "130".
df = pd.read_csv('data.csv')
df.fillna(130,inplace = True)
print(df.to_string())
#Insert the correct argument to make sure that the changes are done for the original DataFrame instead of returning a new one.
df = pd.read_csv('data.csv')
df.dropna(inplace = True)
print(df.to_string())
#Insert the correct syntax for removing duplicates in a DataFrame.
df = pd.read_csv('data.csv')
df.drop_duplicates()
print(df.to_string())

#Pandas Correlation
#Insert a correct syntax for finding relationships between columns in a DataFrame.
df = pd.read_csv('data.csv')
print(df.corr())
#True or false: A correlation of 0.9 is considered a good correletaion.
True
#True or false: A correlation of -0.9 is considered a good correletaion.
#Good Correlation

#Pandas Plotting
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
df.plot()
plt.show()
#Insert the correct syntax for specifying that the plot should be of type 'scatter'.
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
plt.show()
#Insert the correct syntax for specifying that the plot should be of type 'histogram'.
import matplotlib.pyplot as plt
df["Duration"].plot(kind = 'hist')
plt.show()