# RNN Architecture

from keras import Sequential
from keras.layers import Dense,SimpleRNN

model  = Sequential()

model.add(SimpleRNN(3,input_shape=(4,5)))
model.add(Dense(1,activation='sigmoid'))

model.summary()

print(model.get_weights()[1].shape)

model.get_weights()[1]

print(model.get_weights()[2].shape)
model.get_weights()[2]

print(model.get_weights()[3].shape)
model.get_weights()[3]

print(model.get_weights()[4].shape)
model.get_weights()[4]
