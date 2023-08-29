
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

X,y = make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,
                          n_classes=2,n_clusters_per_class=1,random_state=41,hypercube=False,class_sep=15)


plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.show()


def perceptron(X,y):
    w1 =1
    w2=1
    b =1
    lr = 0.1

    for j in range(1000):
        for i in range(X.shape[0]):

            #condtions
            z = w1*X[i][0] + w2*X[i][1] + b

            if z*y[i]<0:
                w1 = w1 + lr*y[i]*X[i][0]
                w2 = w2 + lr*y[i]*X[i][1]
                b = b + lr*y[i]

    return w1,w2,b


w1,w2,b = perceptron(X,y)

print(b)


m = -(w1/w2)
c = -(b/w2)

print(m,c)


X_input = np.linspace(-3,3,100)
y_input = m*X_input+c

plt.plot(X_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,3)
plt.show()

