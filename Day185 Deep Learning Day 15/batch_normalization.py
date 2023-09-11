# Batch Normalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_excel(r'D:\copy of htdocs\practice\Python\200days\Day185 Deep Learning Day 15\concentric_data.xlsx')

df.head()

plt.scatter(df['X'],df['Y'],c=df['class'])

X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

model = Sequential()

model.add(Dense(2,activation='relu',input_dim=2))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history1 = model.fit(X,y,epochs=200,validation_split=0.2)

model = Sequential()

model.add(Dense(3,activation='relu',input_dim=2))
model.add(BatchNormalization())
model.add(Dense(2,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history2 = model.fit(X,y,epochs=200,validation_split=0.2)

plt.plot(history1.history['val_accuracy'],color='black')
plt.plot(history2.history['val_accuracy'],color='green')

