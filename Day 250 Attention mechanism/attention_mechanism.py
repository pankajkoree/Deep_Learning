"""
# Attention mechanism
- The Attention Mechanism
- The General Attention Mechanism
- The General Attention Mechanism with NumPy and SciPy

- Alignment scores :
![image.png](attachment:image.png)

- Weights: ![image-2.png](attachment:image-2.png)

- Context vector:  ![image-3.png](attachment:image-3.png)
"""

import numpy as np
import scipy
import random

# encoder representations of four different words
word_1 = np.array([1, 0, 0])
word_2 = np.array([0, 1, 0])
word_3 = np.array([1, 1, 0])
word_4 = np.array([0, 0, 1])

# generating the weight matrices
np.random.seed(42) # to allow us to reproduce the same attention values
W_Q = np.random.randint(3, size=(3, 3))
W_K = np.random.randint(3, size=(3, 3))
W_V = np.random.randint(3, size=(3, 3))

# generating the queries, keys and values
query_1 = word_1 @ W_Q
key_1 = word_1 @ W_K
value_1 = word_1 @ W_V
 
query_2 = word_2 @ W_Q
key_2 = word_2 @ W_K
value_2 = word_2 @ W_V
 
query_3 = word_3 @ W_Q
key_3 = word_3 @ W_K
value_3 = word_3 @ W_V
 
query_4 = word_4 @ W_Q
key_4 = word_4 @ W_K
value_4 = word_4 @ W_V

# scoring the first query vector against all key vectors
scores = np.array([np.dot(query_1, key_1), np.dot(query_1, key_2), np.dot(query_1, key_3), np.dot(query_1, key_4)])

from scipy.special import softmax


weights = softmax(scores / key_1.shape[0] ** 0.5)

# computing the attention by a weighted sum of the value vectors
attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)
 
print(attention)