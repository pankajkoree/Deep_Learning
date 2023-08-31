import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day175 DL_5\Admission_Predict_Ver1.1.csv')

df.shape

df.head()

df.info()

df.isnull().sum()

df.duplicated().sum()

df.drop(columns=['Serial No.'],inplace=True)

df.head()

X=df.iloc[:,0:-1]
y=df.iloc[:,-1]

print(X)

print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

X_train

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

X_train_scaled= scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

X_train_scaled


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()

model.compile(loss='mean_squared_error',optimizer='Adam')

history=model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)

y_pred=model.predict(X_test_scaled)


from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])