# AGE GENDER

import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


folder_path = r'D:\copy of htdocs\practice\Python\200days\Day195 Deep Learning Day 25\utkface_aligned_cropped\UTKFace'

age=[]
gender=[]
img_path=[]
for file in os.listdir(folder_path):
  age.append(int(file.split('_')[0]))
  gender.append(int(file.split('_')[1]))
  img_path.append(file)


len(age)

df = pd.DataFrame({'age':age,'gender':gender,'img':img_path})

df.shape

df.head()

train_df = df.sample(frac=1,random_state=0).iloc[:100]
test_df = df.sample(frac=1,random_state=0).iloc[100:]


train_df.shape

test_df.shape

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    directory=folder_path,
                                                    x_col='img',
                                                    y_col=['age','gender'],
                                                    target_size=(200,200),
                                                    class_mode='multi_output')

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                    directory=folder_path,
                                                    x_col='img',
                                                    y_col=['age','gender'],
                                                    target_size=(200,200),
                                                  class_mode='multi_output')


from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model

resnet = ResNet50(include_top=False, input_shape=(200,200,3))

resnet = ResNet50(include_top=False, input_shape=(200,200,3))

resnet.trainable=False

output = resnet.layers[-1].output

flatten = Flatten()(output)

dense1 = Dense(512, activation='relu')(flatten)
dense2 = Dense(512,activation='relu')(flatten)

dense3 = Dense(512,activation='relu')(dense1)
dense4 = Dense(512,activation='relu')(dense2)

output1 = Dense(1,activation='linear',name='age')(dense3)
output2 = Dense(1,activation='sigmoid',name='gender')(dense4)

model = Model(inputs=resnet.input,outputs=[output1,output2])


history=model.compile(optimizer='adam', loss={'age': 'mae', 'gender': 'binary_crossentropy'}, metrics={'age': 'mae', 'gender': 'accuracy'},loss_weights={'age':1,'gender':99})

model.fit(train_generator, batch_size=32, epochs=10, validation_data=test_generator)

