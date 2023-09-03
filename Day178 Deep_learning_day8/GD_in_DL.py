# Among batch and Stochastic gradient, which one is faster in terms of execution?


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day178 Deep_learning_day8\Social_Network_Ads.csv')

df.head()

df = df[['Age','EstimatedSalary','Purchased']]

df.head()

X=df.iloc[:,0:2]
y=df.iloc[:,-1]

X
y

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
x_scaled = scalar.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train.shape

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10,activation='relu',input_dim=2))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=10,batch_size=320)

plt.plot(history.history['loss'])

history = model.fit(X_train,y_train,epochs=10,batch_size=1)

plt.plot(history.history['loss'])

