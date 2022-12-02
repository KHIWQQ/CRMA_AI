import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from math import log
data = pd.read_csv('mtcarDataset.csv')


# Homework 1
#1. multi_reg_model1 ==> mpg = m1*wt+ m2*qsec + c1
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt","qsec"]],
                    y = data["mpg"])

X1 = pd.DataFrame({'wt':[4.5, 8 ], 'qsec':[17.7,23.28]})
multi_reg_model1 = multi_reg_model.predict(X1)
print('MPG Predictions from wt & qsec: \n',X1)
print(multi_reg_model1)

# Homework 2
#2. multi_reg_model2 ==> mpg = m1*wt+ m2*qsec+ m3*hp  + c1
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt", "qsec", "hp"]],
                    y = data["mpg"])

X2 = pd.DataFrame({'wt':[4.32, 8], 'qsec':[15,18.7], 'hp':[120,72.8]})
multi_reg_model2 = multi_reg_model.predict(X2)
print('MPG Predictions from wt & qsec & hp: \n',X2)
print(multi_reg_model2)


# Homework 3
#3. multi_reg_model3 ==> mpg = m1*wt+ m2*qsec+ m3*hp+ m4*drat + c1
multi_reg_model = linear_model.LinearRegression()

# Train the model using the mtcars data
multi_reg_model.fit(X = data.loc[:,["wt", "qsec", "hp", "drat"]],
                    y = data["mpg"])

X3 = pd.DataFrame({'wt':[4.32, 8], 'qsec':[14,13.0], 'hp':[120,70.5], 'drat':[3.2,3.5]})
multi_reg_model3 = multi_reg_model.predict(X3)
print('MPG Predictions from wt & qsec & hp & drat: \n',X3)
print(multi_reg_model3)