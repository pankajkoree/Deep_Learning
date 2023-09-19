# CNN Project / Dog VS Cat

#import zipfile

""" 
zip_ref = zipfile.ZipFile(r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\test1.zip")
zip_ref.extractall('/content')
zip_ref.close()
"""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D,Flatten

# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory=r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\dataset\train",
    labels=None,
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    class_names='class_1'
)



validation_ds = keras.utils.image_dataset_from_directory(
    directory=r"D:\copy of htdocs\practice\Python\200days\Day193 Deep Learning Day 23\dataset\test",
    labels=None,
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    class_names='class_2'
)



import tensorflow as tf

# Define your process function
def process(image, labels):
    image = tf.cast(image / 255., tf.float32)
    return image, labels

# Assuming you have defined train_ds and validation_ds as datasets with (image, label) pairs
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)


