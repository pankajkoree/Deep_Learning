# Sentiment Analysis using Integer encoding

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.datasets import imdb
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten

docs = ['go india',
		'india india',
		'hip hip hurray',
		'jeetega bhai jeetega india jeetega',
		'bharat mata ki jai',
		'kohli kohli',
		'sachin sachin',
		'dhoni dhoni',
		'modi ji ki jai',
		'inquilab zindabad']

tokenizer = Tokenizer(oov_token='<nothing>')

tokenizer.fit_on_texts(docs)

tokenizer.word_index

tokenizer.word_counts

tokenizer.document_count

sequences = tokenizer.texts_to_sequences(docs)
print(sequences)

sequences = pad_sequences(sequences,padding='post')

print(sequences)

(X_train,y_train),(X_test,y_test) = imdb.load_data()

X_train[0]

X_train.shape

len(X_train[0])

X_train = pad_sequences(X_train,padding='post',maxlen=50)
X_test = pad_sequences(X_test,padding='post',maxlen=50)

X_train.shape

X_train[0]

model = Sequential()

model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False))
model.add(Dense(1,activation='sigmoid'))

model.summary()

history=model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))


