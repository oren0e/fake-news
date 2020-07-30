import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import calendar

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

# read data
df_fake: pd.DataFrame = pd.read_csv('./data/Fake.csv')
df_true: pd.DataFrame = pd.read_csv('./data/True.csv')

df_fake.head()
df_fake.info()

# create a target variable (fake = 1, true = 0)
df_fake['label'] = 1
df_true['label'] = 0

df = pd.concat([df_fake, df_true], axis=0)
df.head()
del df_true, df_fake
# shuffle the rows
df = df.sample(frac=1, random_state=104).reset_index()
df.head()

# how many subjects are there
df['subject'].value_counts()

# how many topics in fake and true
df.head()
sns.countplot(x='subject', data=df, hue='label')
plt.xticks(rotation=45)
plt.show()

# TODO:
#   1. date should be a numeric field
#   2. subjects should be encoded into categories with a missing category as well for unseen categories
#   3. title and text should participate in the main network procedure.

# fix date
df['date'] = df['date'].str.rstrip()
df['month'] = (df['date'].str.extract(r'(^\w+)', expand=True)
 .replace([month_name for month_name in calendar.month_name], [i for i in range(13)])
.replace([month_name for month_name in calendar.month_abbr], [i for i in range(13)]))
df['day'] = df['date'].str.extract(r'(\d+)', expand=True)
df['year'] = df['date'].str.extract(r'(\d+$)', expand=True)
#df.drop('date', axis=1, inplace=True)
# convert to numeric (day, month, year)

# plots
# construct numeric date for plotting
df_plot = df.copy()
# drop fields that begin with https:// or http:// in the title or text
df_plot.drop(df_plot[df_plot['title'].str.startswith('http')].index, axis=0, inplace=True)
df_plot.drop(df_plot[df_plot['text'].str.startswith('http')].index, axis=0, inplace=True)

df_plot['num_date'] = pd.to_datetime(df_plot[['year','month','day']])  # raises error!
print(df_plot.iloc[[13295]])
df_plot.loc[[13325]]

df.isnull().sum()
df_plot.isna().sum()

df_plot.drop(13325, axis=0, inplace=True)
df_plot['num_date'] = pd.to_datetime(df_plot[['year','month','day']])
df_plot.head()
df_plot.info()

df_plot['year'] = pd.to_numeric(df_plot['year'])
df_plot['year'].describe()
df_plot['year'].value_counts()
df_plot[df_plot['year'] == 18]
df_plot.loc[df_plot['year'] == 18, 'year'] = 2018

df_plot['month'] = pd.to_numeric(df_plot['month'])
df_plot.loc[df_plot['month'] > 12,:]
df_plot.loc[df_plot['month'] > 12, 'month'] = 2
df_plot['month'].value_counts().reset_index().sort_values(by='index')

df_plot['num_date'] = pd.to_datetime(df_plot[['year','month','day']])
df_plot.head()
df_plot['num_date'].describe()


df_prog_label = df_plot.groupby('num_date')['label'].apply(lambda x: sum(x)/len(x)).reset_index().sort_values(by='label')
sns.lineplot(x='num_date', y='label', data=df_prog_label)
plt.show()
df_prog_label = df_plot.groupby('num_date')['label'].apply(lambda x: sum(x==1)).reset_index().sort_values(by='label')
df_prog_label
sns.lineplot(x='num_date', y='label', data=df_prog_label)
plt.show()

# save df_plot to pickle
pd.to_pickle(df_plot, './data/df.pkl')