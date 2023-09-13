# Netsro Accelertaed Gradient

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day186 Deep Learning Day 16\DailyDelhiClimateTrain.csv')

df = df[['date','humidity']]

print(df)

plt.figure(figsize=(18,9))
plt.scatter(df['date'],df['humidity'])


weighted_estimated= df['humidity'].ewm(alpha=0.9).mean()

print(weighted_estimated)

df['ewm'] = weighted_estimated

print(df)

plt.figure(figsize=(18,9))
plt.scatter(df['date'],df['humidity'])
plt.plot(df['date'],df['ewm'],'g-')

