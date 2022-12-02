import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from math import log
data = pd.read_csv('mtcarDataset.csv')
print(type(data))
print(data.head())
print(data.keys())

#step3
data.plot(kind="scatter",x="wt",y="mpg",figsize=(9,9),color="black",title="scatter plot of wt vs mpg.")
plt.show()

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
ax1.set_title("MPG Boxplot")
ax1.boxplot(data['mpg'],labels=['mpg'])


ax2.set_title("WT Boxplot")
ax2.boxplot(data['wt'],labels=['wt'])
#qsec

ax3.set_title('HP Boxplot')
ax3.boxplot(data['hp'], labels=['hp'])
#qsec

ax4.set_title('Qsec Boxplot')
ax4.boxplot(data['qsec'], labels=['qsec'])
#Set a tight layout

plt.tight_layout()
plt.show()
#Density plot
#strp3
pltl=pd.DataFrame(data[ 'mpg'])
pltl.plot(kind="density",title='Density plot of mpg');
plt2=pd.DataFrame(data[ 'wt'])
plt2.plot(kind="density",title='Density plot of wt');
plt3=pd.DataFrame(data['hp'])

plt3.plot(kind="density", title='Density plot of hp');
plt4=pd.DataFrame(data['qsec'])
plt4.plot(kind="density",title='Density plot of qsec');
plt.show()

#Corellation matrix
print("Correlation matrix")
corrMatrix = data.corr()
print(corrMatrix )
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Linear regression 1s used to fit the best line to the data
print("####sesl Sinple Linear Regression##sussis")
# Initialize model
regression_model = linear_model.LinearRegression()
# Train the model using the mtcars data
regression_model.fit(X = pd.DataFrame(data["wt"]),
y = data["mpg"])
# Check trained model y-intercept
print(regression_model.intercept_)
# Check trained model coefficients
print(regression_model.coef_)
#It can be seen that the y-intercept term is set to 37.2851 and the coef
# making the equation => mpg = 37.2851 - 5.3445 * wt.
# Check R-squared
score = regression_model.score(X = pd.DataFrame(data["wt"]),y = data["mpg"])
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
WT2 = pd.DataFrame([4.32, 6.55])
print(WT2)
predicted_MPG2 = regression_model.predict(WT2)
print(predicted_MPG2)
print("Predicted MPG of WT = {0} is {1}".format(WT2[0][0], predicted_MPG2[0]))
print("Predicted MPG of WT = {0} is {1}".format(WT2[0][1], predicted_MPG2[1]))

#Prediction from wt values
train_prediction = regression_model.predict(X = pd.DataFrame(data["wt"]))

# Actual - prediction = residuals
residuals = data["mpg"] - train_prediction
print(residuals.describe())

data.plot(kind="scatter",
          x="wt",
          y="mpg",
          figsize=(9,9),
          color="black",
          xlim = (0,7),
          title='A fitted linear line of function wt and mpg')

# plot regression line
plt.plot(data["wt"],    # Explanitory variable
         train_prediction,  # Predicted values
         color="blue")
plt.show()

#Step 6 :Model Evaluation
#R-squared value
print("R-suared value: ")
print(score)
#Calculate MAE, MSE, RMSE
y_true = data["mpg"]
y_pred = train_prediction
print("Calcuated MAE, MSE, RMSE:")
print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

#Calculating AIC
# number of parameters
num_params = len(regression_model.coef_) + 1
n = len(y_true)
mse = metrics.mean_squared_error(y_true, y_pred)

print("AIC:")
aic = n * log(mse) + 2 * num_params
print(aic)

print("BIC:")
bic = n * log(mse) + num_params * log(n)
print(bic)

#Step 7: ML Modeling of Multiple Linear Regression
#When more X variables are added, a multiple linear regression model is used.
#The multiple linear regression model produced is
#mpg = 37.22727011644721 -3.87783074wt -0.03177295hp

print('######2 Multiple Linear Regression######')
# Initialize model
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt","hp"]],
                    y = data["mpg"])

# Check trained model y-intercept
print("Y-intercept and slop: ")
print(multi_reg_model.intercept_)

# Check trained model coefficients (scaling factor given to "wt")
print(multi_reg_model.coef_)

#Check R-squared
print("R-squared value: ")
score = multi_reg_model.score(X = data.loc[:,["wt","hp"]],
                              y = data["mpg"])
print(score)

#Prediction from wt and hp values
# X1 = pd.DataFrame({'wt':[4.32, 8], 'hp':[120,175]})
# multi_reg_model_predictions = multi_reg_model.predict(X1)
# print('MPG Predictions from wt & hp: \n',X1)
# print(multi_reg_model_predictions)

# Homework 1
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt","qsec"]],
                    y = data["mpg"])

X1 = pd.DataFrame({'wt':[4.5, 8 ], 'qsec':[17.7,23.28]})
multi_reg_model1 = multi_reg_model.predict(X1)
print('MPG Predictions from wt & qsec: \n',X1)
print(multi_reg_model1)

# Homework 2
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt", "qsec", "hp"]],
                    y = data["mpg"])

X2 = pd.DataFrame({'wt':[4.32, 8], 'qsec':[15,18.7], 'hp':[120,72.8]})
multi_reg_model2 = multi_reg_model.predict(X2)
print('MPG Predictions from wt & qsec & hp: \n',X2)
print(multi_reg_model2)


# Homework 3
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt", "qsec", "hp", "drat"]],
                    y = data["mpg"])

X3 = pd.DataFrame({'wt':[4.32, 8], 'qsec':[14,13.0], 'hp':[120,70.5], 'drat':[3.2,3.5]})
multi_reg_model3 = multi_reg_model.predict(X3)
print('MPG Predictions from wt & qsec & hp & drat: \n',X3)
print(multi_reg_model3)