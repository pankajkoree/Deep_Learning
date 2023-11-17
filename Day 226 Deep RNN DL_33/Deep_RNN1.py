# Deep RNN 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense

# laoding the dataset
num_words = 10000
(X_train,y_train), (X_test,y_test) = imdb.load_data(num_words=num_words)

# pad sequences
maxlen = 100
X_train = pad_sequences(X_train,maxlen=maxlen)
X_test = pad_sequences(X_test,maxlen=maxlen)

# bulding the bidirectional model
Embedding_dim = 32

model = Sequential([
    Embedding(input_dim=num_words, output_dim = Embedding_dim,input_length=maxlen),
    SimpleRNN(5, return_sequences=True),
    SimpleRNN(5),
    Dense(1, activation='sigmoid')
])

model.summary()

# compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.2)

