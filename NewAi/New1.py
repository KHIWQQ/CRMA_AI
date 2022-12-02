import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from math import log

# Loading mtcars dataset and printing information about it
data = pd.read_csv('mtcarDataset.csv')
# print(type(data))
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# # Printing keys
# print(data.keys())

#Scatter plot
data.plot(kind="scatter",
            x="wt",
            y="mpg",
            figsize=(9,9),
            color="black",
            title='Scatter plots of wt vs. mpg');
# Showing plot
plt.show()

# Box plots
#4x4 Layouts of plot ax1, ax2, ax3, ax4
fig, ((ax1, ax2),(ax3, ax4))=plt.subplots(nrows=2, ncols=2)

#mpg
ax1.set_title('MPG Boxplot')
ax1.boxplot(data['mpg'], lables=['mpg'])

#wt
ax2.set_title('Wt Boxplot')
ax2.boxplot(data['wt'], lables=['wt'])

#qsec
ax3.set_title('HP Boxplot')
ax3.boxplot(data['hp'], labels=['hp'])

#qsec
ax4.set_title('Qsec Boxplot')
ax4.boxplot(data['qsec'], lables=['qsec'])

#Set a tight layout
plt.tight_layout()
plt.show()

#Density plot
plt1=pd.DataFrame(data['mpg'])
plt1.plot(kind="density",title='Density plot of mpg');

plt2=pd.DataFrame(data['wt'])
plt2.plot(kind="density",title='Density plot pf wt');

plt3=pd.DataFrame(data['hp'])
plt3.plot(kind="density",title='Density plot of hp');

plt4=pd.DataFrame(data['qsec'])
plt4.plot(kind="density",title='Density plot of qsec');
plt.show()

#Corllation matrix
print("Correlation matrix")
corrMatrix = data.corr()
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()

# Step 4:ML Modeling using a Simple Linear Regression method
#Linear regression is used to fit best line to the data
print('######1 Simple Linear Regression######')
# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the mtcars data
regression_model.fit(X = pd.DataFrame(data["wt"]),
                     Y = data["mpg"])

# Cheack trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(regression_model.coef_)

#It can be seen that the y-intercept term is set to37.2851 and the coefficient for the weight variable is -5.3445,
# making the equation => mpg = 37.2851 - 5.3445 * wt.

# Check R-squared
score = regression_model.score(X = pd.DataFrame(data["wt"]),
                               Y = data["mpg"])
print("R-squared value: ")
print(score)

#Step 5:Making predictions
print("###Prediction 1 ###")
WT1 = pd.DataFrame([3.52])
print(WT1)
predicted_MPG = regression_model.predict(WT1)
print(predicted_MPG)
print("Predicted MPG of WT = {0} is {1}".format(WT1[0][0], predicted_MPG[0]))

print("###Prediction 2###")
WT2 = pd.Dataframe([4.32, 6.55])
print(WT2)
predicted_MPG2 = regression_model.predict(WT2)
print(predicted_MPG2)
print("Predicted MPG of WT = {0} is {1}".format(WT1[0][1], predicted_MPG[0]))
print("Predicted MPG of WT = {0} is {1}".format(WT1[0][1], predicted_MPG[1]))

#Prediction from wt values
train_prediction = regression_model.predict(X = pd.DataFrame(data["wt"]))
