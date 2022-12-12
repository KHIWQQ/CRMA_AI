import pandas as pd
from tensorflow import keras
import numpy as np
<<<<<<< HEAD:KNN/ANN.py
=======
import matplotlib.pyplot as plt
>>>>>>> e5dd0afd230d170c616f6eb7d5f9c44dfdaf82a9:ANN/ANN.py
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# load the dataset
dataset = np.loadtxt('ABC.csv', delimiter=',')
<<<<<<< HEAD:KNN/ANN.py
# dataset = pd.read_csv('Test.csv')
# split into input (X) and output (y) variables
# X = dataset
# Y = dataset
=======

# split into input (X) and output (y) variables
>>>>>>> e5dd0afd230d170c616f6eb7d5f9c44dfdaf82a9:ANN/ANN.py
X = dataset[:,0:8]
Y = dataset[:,8]
print(X)
print(Y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
<<<<<<< HEAD:KNN/ANN.py
# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=100)
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
# accuracy = model.evaluate(X. Yverbose=0) tin case we do not need to  print out
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = model.predict(X)
=======

# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=100)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)

# accuracy = model.evaluate(X. Yverbose=0) tin case we do not need to  print out
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict(X)

>>>>>>> e5dd0afd230d170c616f6eb7d5f9c44dfdaf82a9:ANN/ANN.py
# round predictions
rounded = [round(x[0]) for x in predictions]