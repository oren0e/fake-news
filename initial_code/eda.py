import pandas as pd
import numpy as np

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
df = df.sample(frac=1).reset_index()
df.head()

# how many subjects are there
df['subject'].value_counts()

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
df.drop('date', axis=1, inplace=True)
