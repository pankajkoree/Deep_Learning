# Using Relu for weight initialization problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r'D:\copy of htdocs\practice\Python\200days\Day184 Dee[ Learning Day 14\dataset.xlsx')

df.head()

plt.scatter(df['X'],df['Y'],c=df['class'])

X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10,activation='relu',input_dim=2))
model.add(Dense(1,activation='relu'))

model.summary()


# Set parameters to 0
model.get_weights()

initial_weights = model.get_weights()

initial_weights[0] = np.zeros(model.get_weights()[0].shape)
initial_weights[1] = np.zeros(model.get_weights()[1].shape)
initial_weights[2] = np.zeros(model.get_weights()[2].shape)
initial_weights[3] = np.zeros(model.get_weights()[3].shape)

model.set_weights(initial_weights)

model.get_weights()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X,y,epochs=100,validation_split=0.2)

model.get_weights()

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X,y.astype('int'), clf=model, legend=2)

