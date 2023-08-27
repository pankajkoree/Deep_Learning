import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day172 DL_Perceptron\placement.csv')

print(df.shape)

print(df.head())

sns.scatterplot(x=df['cgpa'],y=df['resume_score'],hue=df['placed'])
plt.show()


X= df.iloc[:,0:2]
y = df.iloc[:,-1]


from sklearn.linear_model import Perceptron

p=Perceptron()


print(p.coef_)

print(p.intercept_)

from mlxtend.plotting import plot_decision_regions


plot_decision_regions(X.values,y.values,clf=p,legend=2)
plt.show()
