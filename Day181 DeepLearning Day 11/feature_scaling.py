import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day181 DeepLearning Day 11\Social_Network_Ads.csv')

df = df.iloc[:,2:]
df.head()

sns.scatterplot(x=df.iloc[:,0],y=df.iloc[:,1])
plt.show()

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


model = Sequential()

model.add(Dense(128,activation='relu',input_dim=2))
model.add(Dense(1,activation='sigmoid'))


model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)


import matplotlib.pyplot as plt
plt.plot(history.history['val_accuracy'])
plt.show()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled)


sns.scatterplot(x=X_train_scaled[:,0],y=X_train_scaled[:,1])
plt.show()

model = Sequential()

model.add(Dense(128,activation='relu',input_dim=2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train_scaled,y_train,validation_data=(X_test_scaled,y_test),epochs=100)



plt.plot(history.history['val_accuracy'])
plt.show()