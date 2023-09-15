# Keras Tuner / Hyperparameter Tuning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

df=pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day189 Deep Learning Day 19\diabetes.csv')

print(df.head())

print(df.corr()['Outcome'])

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X=scalar.fit_transform(X)

print(X.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=8))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=100,validation_data=(X_test,y_test))

# how to select appropriate optimizer
import keras_tuner as kt

def bulid_model(hp):
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim=8))
    model.add(Dense(1,activation='sigmoid'))

    optimizer=hp.Choice('optimizer',values=['adam','sgd','rmsprop','adadelta'])

    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(bulid_model,
                        objective='val_accuracy',
                        max_trials=5)


tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

tuner.get_best_hyperparameters()[0].values

model =  tuner.get_best_models(num_models=1)[0]

print(model.summary())

model.fit(X_train,y_train,batch_size=32,epochs=100,initial_epoch=6,validation_data=(X_test,y_test))

# to get right number of neurons
def bulid_model(hp):
    model = Sequential()

    units = hp.Int('units',min_value=8,max_value=128)

    model.add(Dense(units=units,activation='relu',input_dim=8))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    return model


tuner = kt.RandomSearch(bulid_model,
                        objective='val_accuracy'
                        ,max_trials=5,
                        directory='mydir',project_name='Pankaj')


tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

tuner.get_best_hyperparameters()[0].values

model = tuner.get_best_models(num_models=1)[0]

model.fit(X_train,y_train,batch_size=32,epochs=100,initial_epoch=6)

# how to select no of layers

def bulid_model(hp):
    model = Sequential()
    model.add(Dense(72,activation='relu',input_dim=8))

    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        
        model.add(Dense(72,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

        return model
    



tuner=kt.RandomSearch(bulid_model,
                      objective='val_accuracy',
                      max_trials=3,
                      directory='mydir',
                      project_name='num_layers')


tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))


tuner.get_best_hyperparameters()[0].values

model=tuner.get_best_models(num_models=1)[0]

model.fit(X_train,y_train,epochs=100,initial_epoch=6,validation_data=(X_test,y_test))

# for both layers and nodes
def bulid_model(hp):
    model =Sequential()
    counter= 0

    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        if counter == 0:
            model.add(Dense(
                hp.Int('units'+str(i),min_value=8,max_value=128,step=8),
                activation=hp.Choice('activation'+str(i),values=['relu','tanh','sigmoid']),
                input_dim=8
                ))
            model.add(Dropout(hp.Choice('droput'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        else:
            model.add(Dense(
                hp.Int('units'+str(i),min_values=8,max_value=128,step=8),
                activation=hp.Choice('activation'+str(i),values=['relu','tanh','sigmoid'])
                ))
            model.add(Dropout(hp.Choice('droput'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        counter+=1

        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer=hp.Choice('optimier',values=['rmsprop','adam','sgd','nadam','adadelta']),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model
            


tuner = kt.RandomSearch(bulid_model,
                        objective='val_accuracy',
                        max_trials=3,
                        directory='mydir',
                        project_name='final1')


tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

tuner.get_best_hyperparameters()[0].values

model = tuner.get_best_models(num_models=1)[0]

model.fit(X_train,y_train,epochs=200,initial_epoch=5,validation_data=(X_test,y_test))

