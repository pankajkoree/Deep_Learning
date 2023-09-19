# Data Augmentation

import tensorflow
from tensorflow import _keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

img = image.load_img(r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\img\cat.18.jpg",target_size=(200,200))

import matplotlib.pyplot as plt

plt.imshow(img)

type(img)

datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='reflect'   #constant can be used for black background around it
)

img = image.img_to_array(img)

img.shape

input_batch = img.reshape(1,200,200,3)

i=0
for output  in datagen.flow(input_batch,batch_size=1,save_to_dir='aug'):
    i = i+1
    if i==10:
        break


input_batch.shape

