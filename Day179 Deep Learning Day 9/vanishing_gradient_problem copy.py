# Using  Sigmoid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential

X,y = make_moons(n_samples=250, noise=0.05, random_state=42)

plt.scatter(X[:,0],X[:,1], c=y, s=100)
plt.show()

model = Sequential()

model.add(Dense(10,activation='sigmoid',input_dim=2))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.get_weights()[0]

old_weights = model.get_weights()[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


model.fit(X_train, y_train, epochs = 100)

new_weights = model.get_weights()[0]

model.optimizer.get_config()["learning_rate"]

gradient = (old_weights - new_weights)/ 0.001
percent_change = abs(100*(old_weights - new_weights)/ old_weights)

print(gradient)

print(percent_change)


print(old_weights)

print(new_weights)

