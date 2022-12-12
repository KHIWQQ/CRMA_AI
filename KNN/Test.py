import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('ABC.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
print(X)
print(y)
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #hidden
model.add(Dense(8, activation='relu')) #input
model.add(Dense(1, activation='sigmoid')) #output

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy' ])
# fit the keras model on the dataset
model.fit(X, y, epochs=250, batch_size=100)
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
_, accuracy = model.evaluate(X, y, verbose=0) #in case we do not need to print out
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in  predictions]
for i in range(100):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
