# NEWSWIRES CLASSIFICATION - a multiclass classification using Reuters Dataset

"""
### Multiclass classification
- Having imbalanced dataset
- more than two classes
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#- loading the dataset

from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)  # restricting the data to 10000 most frequently occuring

# checking length of trainig and test data
print(len(train_data))

print(len(test_data))

len(train_data[10])

print(train_data[10])

# decoding newswires back to text

word_index = reuters.get_word_index()

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_newswire = ''.join([reversed_word_index.get(i-3,'?') for i in train_data[0]])   # note that the indices are offset by 3 coz 0,1 and 2 are reveresed indices for "padding","start of sequence","unknown"

train_labels[10]

# preparing the data

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    
    for i, sequence  in enumerate(sequences):
        results[i,sequence] = 1.
    return results

X_train = vectorize_sequences(train_data)  # vectorizing training data
X_test = vectorize_sequences(test_data)     # vectorizing test data

# encoding the data

def one_to_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))

    for i,label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels=one_to_hot(train_labels)
one_hot_test_labels = one_to_hot(test_labels)

# built in way :
from tensorflow.keras.utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)

one_hot_test_labels = to_categorical(test_labels)

# - building the network

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

# compiling the model

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# validating the approach

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# training the model
history=model.fit(partial_X_train,
          partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(X_val,y_val))

# plotting the model

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'b+',label='Training label')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'b+',label='Traing accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validaion accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

"""
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # Adjust input shape to match your data

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""

# retraining the model from scratch

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # Adjust input shape to match your data

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_X_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(X_val,y_val))

# evaluating the model

results = model.evaluate(X_test,one_hot_test_labels)

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)

float(np.sum(hits_array))/len(test_labels)

# generating predictions on new data

predictions = model.predict(X_test)

predictions[0].shape

# coffiecients in this vector sum to 1
np.sum(predictions[0])

# a different way to handle labels and loss

y_train=np.array(train_labels)
y_test=np.array(test_labels)

# changing the loss function

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
               metrics=['acc'])

# modeling with an information bottleneck

model = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_X_train,
          partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(X_val,y_val))