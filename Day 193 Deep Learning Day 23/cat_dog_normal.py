# Cat Dog Normal without Data Augmentation

import cv2
import os
import random
import numpy as np

mydir =  r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\data\train"

categories = ['cat1','dog1']

data = []
for i in categories:
    folder_path = os.path.join(mydir,i)
    
    if i=='cat1':
        label = 0
    else:
        label = 1

        for j in os.listdir(folder_path):
            img_path = os.path.join(folder_path,j)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(150,150))
            data.append([img,label])


random.shuffle(data)

X=[]
y=[]
for i in data:
    X.append(i[0])
    y.append(i[1])

y = np.array(y)

X = np.array(X)

X.shape

X = X/255

# CNN 
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation,Dropout

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X,y,epochs=5,validation_split=0.1)

