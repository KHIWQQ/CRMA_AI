import pandas as pd
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
dataset = pd.read_csv('mtcarDataset.csv')

#Step 8: Excercise - (15 points)
#Create  linear multiple regression  to predict mpg from:
#1. multi_reg_model1 ==> mpg = m1*wt+ m2*qsec + c1
#2. multi_reg_model2 ==> mpg = m1*wt+ m2*qsec+ m3*hp  + c1
#3. multi_reg_model3 ==> mpg = m1*wt+ m2*qsec+ m3*hp+ m4*drat + c1

# print(dataset.head())

x = dataset[['wt','qsec']]
y = dataset['mpg']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

mlr = linear_model.LinearRegression()
mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff.head())


meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

#----------------------------------------------------------------


# x = dataset[['wt','qsec','hp']]
# y = dataset['mpg']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

# mlr = linear_model.LinearRegression()
# mlr.fit(x_train, y_train)

# print("Intercept: ", mlr.intercept_)
# print("Coefficients:")
# print(list(zip(x, mlr.coef_)))

# #Prediction of test set
# y_pred_mlr= mlr.predict(x_test)
# #Predicted values

# mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
# print(mlr_diff.head())


# meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
# meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
# rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
# print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
# print('Mean Absolute Error:', meanAbErr)
# print('Mean Square Error:', meanSqErr)
# print('Root Mean Square Error:', rootMeanSqErr)


#----------------------------------------------------------------

# x = dataset[['wt','qsec','hp','drat']]
# y = dataset['mpg']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#
# mlr = linear_model.LinearRegression()
# mlr.fit(x_train, y_train)
#
# print("Intercept: ", mlr.intercept_)
# print("Coefficients:")
# print(list(zip(x, mlr.coef_)))
#
# #Prediction of test set
# y_pred_mlr= mlr.predict(x_test)
# #Predicted values
# print("Prediction for test set: {}".format(y_pred_mlr))
# mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
# print(mlr_diff.head())
#
#
# meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
# meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
# rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
# print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
# print('Mean Absolute Error:', meanAbErr)
# print('Mean Square Error:', meanSqErr)
# print('Root Mean Square Error:', rootMeanSqErr)