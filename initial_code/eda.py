import pandas as pd
import numpy as np

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

# read data
df_fake: pd.DataFrame = pd.read_csv('../data/Fake.csv')
df_true: pd.DataFrame = pd.read_csv('../data/True.csv')

df_fake.head()
df_fake.info()

# create a target variable (fake = 1, true = 0)
df_fake['label'] = 1
df_true['label'] = 0

df = pd.concat([df_fake, df_true], axis=0)
df.head()

# shuffle the rows
df = df.sample(frac=1)
df.head()

# how many subjects are there
df['subject'].value_counts()

# TODO:
#   1. date should be a numeric field
#   2. subjects should be encoded into categories with a missing category as well for unseen categories
#   3. title and text should participate in the main network procedure.

