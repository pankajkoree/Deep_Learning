"""import numpy as mp
import pandas  as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
df=pd.read_excel(r'D:\copy of htdocs\practice\Python\200days\Day178 Deep_learning_day8\find_s_dataset_for_poisonous.xlsx')

print(df)

num = LabelEncoder()

train = df.drop('Poisonous',axis='columns')

print(train)

target=df['Poisonous']

print(target)

train['color_n'] = num.fit_transform(train['Color'])
train['toughness_n'] = num.fit_transform(train['Toughness'])
train['fungus_n'] = num.fit_transform(train['Fungus'])
train['appearance_n'] = num.fit_transform(train['Appearance'])

print(train)


train_encoded = pd.get_dummies(train)

clf = GaussianNB()

clf.fit(train_encoded,target)

sample = {
    'color_n':[1],
    'toughness_n':[1],
    'fungus_n':[0],
    'appearance_n':[0],
    'Color_green':[1],
    'Color_brown':[1],
    'Color_orange':[0],
    'Toughness_soft':[0],
    'Toughness_hard':[1],
    'Fungus_yes':[1],
    'Fungus_no':[0],
    'Appearance_wrinkled':[1],
    'Appearance_smooth':[0]
}

clf.score(train_encoded,target)

sample_df = pd.DataFrame(sample)

clf.predict(sample_df)"""

