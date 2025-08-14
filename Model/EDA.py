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
features = ['Age','Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview' ]
columns_to_plot = features
for col in columns_to_plot:
    ct = pd.crosstab(df[col],df['treatment'])
    ct.plot(kind='bar',stacked =True)
    plt.title(f'Treatment vs {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Count')
    plt.legend(title="Treatment")
    plt.tight_layout()
    plt.show()
