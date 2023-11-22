# Hold out validation

# importing the modules
import numpy as np

num_validation_samples = 10000

np.random.shuffle(data) # shuffling the data is usually appropriate

validation_data = data[:num_validation_samples] # defines the validation set
data = data[num_validation_samples:]

training_data - data[:]

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# retrain and evaluation
model = get_model()
model.train(np.concatenate([training_data,
                            validation_data]))

test_score = model.evaluate(test_data)