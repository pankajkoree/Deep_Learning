import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day174 Customer Churn Prediction\Churn_Modelling.csv')

df.head()

df.info()

df.duplicated().sum()

df['Exited'].value_counts()
df['Geography'].value_counts()


df['Gender'].value_counts()

df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)

df.head()

df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)

from sklearn.model_selection import train_test_split


X=df.drop(columns=['Exited'])
y = df['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train.shape

X.head()

y.head()

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

X_train_scaled

import tensorflow
from tensorflow import keras
from tensorflow.keras import  Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=11))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

history=model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)

model.summary()

model.layers[0].get_weights()

model.layers[1].get_weights()

model.layers[2].get_weights()

y_log = model.predict(X_test_scaled)


y_pred=np.where(y_log>0.5,1,0)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

history.history

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()