# Transfer Learning Feature Extraction Without Data Augmentation

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
import cv2
import os
import random
import numpy as np
from keras.applications.vgg16 import VGG16


mydir =  r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\dataset"

conv_base = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape=(150,150,3)
)


conv_base.summary()

model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.summary()

conv_base.trainable = False

# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory = r'D:\copy of htdocs\practice\Python\200days\Day194 Deep Learning Day 24\train',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(150,150)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = r'D:\copy of htdocs\practice\Python\200days\Day194 Deep Learning Day 24\test',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(150,150)
)

# Normalize
def process(image,label):
    image = tensorflow.cast(image/255. ,tensorflow.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=1,validation_data=validation_ds)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

