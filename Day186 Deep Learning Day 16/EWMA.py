# Exponentially Weighted Moving Avergae

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day186 Deep Learning Day 16\DailyDelhiClimateTest.csv')

df = df[['date','meantemp']]

print(df)

plt.scatter(df['date'],df['meantemp'])

x1= df['meantemp'].ewm(alpha=0.9).mean()

print(x1)

df['ewma'] = x1

print(df)

plt.scatter(df['date'],df['meantemp'])
plt.plot(df['date'],x1,label='alpha=0.9',color='black')
plt.legend()
plt.show()

