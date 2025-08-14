import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
df = pd.read_csv('survey.csv')
for i in df.columns:
    print(i,df[i].isna().sum())
df.shape
df.info()
df.head(15)
df['Age'].describe()
df['Age'].sort_values(ascending=False).head(10)
for i in df.columns:
    if i!='Age':
        print(i,df[i].unique())
