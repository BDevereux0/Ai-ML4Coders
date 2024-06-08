import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#define neural network. Sequential API used to build NN.
#Dense = layer type. units = neurons per layer, input shape = type of data, but not sure on this one.
#The 2nd layer doesn't require input_shape because it is implictly added.
model = Sequential([Dense(units=1, input_shape=[1])],
                   [Dense(units=1,)])

#Optimizer will make better guesses based off of loss function. Loss evaluates whether guess is close or far.
model.compile(optimizer='sgd', loss='mean_squared_error')

#data sets
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0,5.0,7.0])

#
model.fit(xs, ys, epochs=500)

numAr = np.array([[10.0]])
print(model.predict(numAr))