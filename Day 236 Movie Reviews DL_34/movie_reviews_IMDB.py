# Movie Review Classification - A Binary Classification using IMDB Dataset

"""
### Two class classification or Binary classification
- It is the most widely applied kind of machine learning problem.
- Here, we'll classify movie reviews as positive or negative, based on the text of the reviews

"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the IMDB Dataset
from keras.datasets import imdb

# the num_words is used to set the number of datas that occur most frequently, so that the system can handle it effectively
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])

train_data[0]

# set of reviews : output column
train_labels

n = int(input("Index no: "))
train_labels[n]

# as we restricetd the data, lets check whether it exceeds the limit?
max([max(sequence) for sequence in train_data])

# quick decoding the selected reviews back to english
word_index = imdb.get_word_index()  #word_index is a dictionary mapping words to integer index 

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])  # reverse it, mapping integer indices to words

decoded_review = ''.join(
    [reverse_word_index.get(i -3, '?') for i in train_data[0]]     # decodes the reviews. Note that the indices are offset by 3 because 0,1 and 2 are reveresed indices for "padding","start of sequence", and "unknown"
)

"""
### preparing the datas
- Words can't be feed into Neural network, we need to convert it into tensors.
- there's two types 
- 1. padding the lists - so that all of the datas have same length
- 2. One hot encoding - converts all datas into 0s and 1s
"""

# preparing the datas
# lets gi with the latter solution to vectorize the data, which will be more easy to understand

def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))    # creates all zero matrix of shape(len(sequences), dimensions)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.   # sets specific indices of results[i] to 1s
    return results

X_train = vectorize_sequences(train_data)   # vectorizing training data
X_test = vectorize_sequences(test_data)     # vectorizing test data

# let's check how it looks
X_train[0]

# vectorizing labels, straightforward
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# now the datas can be fed into neural network

### building the neural network

# defining the model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# compiling the models

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',   # binary_crossentropy isn't a viable  choice but its a best choice while dealing with the models that output probabilities
              metrics=['accuracy'])

# configuring the optimizer
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# using custom losses and metrics

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# validating the approach
x_val = X_train[:10000]
partial_x_train = X_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# training the model

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# fitting the models
history=model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_val,y_val)
)

history_dict = history.history

history_dict.keys()

# plotting the training and validation loss

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs=range(1,len(history_dict['accuracy'])+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')    # bo is for 'blue dot'
plt.plot(epochs,val_loss_values,'b',label='Validation loss')    # b is for solid blue line
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.show()

# plotting the training and validation accuracy

plt.clf()   # clears the figure

acc_values = history_dict['accuracy']
val_acc_loss = history_dict['val_accuracy']

plt.plot(epochs,acc_values,'bo',label='Training accuracy')
plt.plot(epochs,val_acc_loss,'b',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Retraining a model from scratch
model = models.Sequential()

model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(1,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,epochs=4,batch_size=512)
results = model.evaluate(X_test,y_test)

print(results)

# using trained network to generate prediction on new data

model.predict(X_test)