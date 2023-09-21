# Function API


from keras.models import Model
from keras.layers import *

# Define the input tensor
x = Input(shape=(3,))

# Define the layers
hidden1 = Dense(128, activation='relu')(x)
hidden2 = Dense(64, activation='relu')(hidden1)

output1 = Dense(1, activation='linear')(hidden2)
output2 = Dense(1, activation='sigmoid')(hidden2)

# Create the model
model = Model(inputs=x, outputs=[output1, output2])

# Summary and plot the model
model.summary()
from keras.utils import plot_model
plot_model(model, show_shapes=True)
