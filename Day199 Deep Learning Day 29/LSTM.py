# Long Short Term Memory

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv(r"D:\copy of htdocs\practice\Python\200days\Day199 Deep Learning Day 29\airline-passengers.csv")

print(df.shape)

df =  df['Passengers']

plt.plot(df)
plt.show()

tf.random.set_seed(7001)


df = df.values
df = df.astype('float')


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

from sklearn.preprocessing import StandardScaler

# Assuming df is a one-dimensional array
df = df.reshape(-1, 1)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


train_size = int(len(df)*0.67)
test_size = len(df) - train_size

train,test = df[0:train_size,:],df[train_size:len(df),:1]


def cr_df(df,look_back=1):
    dfX, dfY = [],[]

    for i in range(len(df)-look_back-1):
        a = df[i:(i+look_back),0]
        dfX.append(a)
        dfY.append(df[i+look_back,0])
    return np.array(dfX),np.array(dfY)


look_back = 1
trainX, trainY = cr_df(train, look_back)
testX, testY = cr_df(test,look_back)

trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))

testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(4,input_shape=(1,look_back)))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX,trainY, epochs=10,batch_size=1,verbose=2)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict =  scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

from sklearn.metrics import mean_squared_error

trainscore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print("Train score : %.2f RMSE"%(trainscore))

testscore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print("Train score : %.2f RMSE"%(testscore))


plt.plot(df,color='blue')
plt.plot(testPredict,color='orange')


