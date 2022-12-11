# import numpy
import pandas as pd
# import keras.models
# import sequential
# import keras. layers
# # import dense
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# load the dataset
dataset = pd.read_csv('pima-indians-diabetes', detimiter=',')
# split into input (X) and output (y) variables
X = dataset[: ,0:8]
y = dataset[: ,8]
print (X)
print(y)
# define the keras model
model = sequential()
model.add(dense(12, input_din=8, activation='relu'))
model.add(dense(8, activation='relu'))
model.add(dense(1, activation='sigmoid'))
# compile the keras model
model.compile (Loss='binary_crossentropy', optinizer='adam', netrics=['accuracy' ])
# fit the keras model on the dataset
model.fit(X, y, epochs=250, batch_size=100)
# evaluate the keras model
_, accuracy = model.evaluate (X, y)
# accuracy = model.evaluate(X. Yverbose=0) tin case we do not need to  print out
print ('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = model.predict_classes (X)