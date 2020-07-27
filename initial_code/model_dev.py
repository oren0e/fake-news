import pandas as pd
import numpy as np

from typing import Tuple

from sklearn.model_selection import train_test_split

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

SEED = 104

# read the data
df: pd.DataFrame = pd.read_pickle('./data/df.pkl')

# remove unused variables
df.drop(['date', 'num_date', 'index'], axis=1, inplace=True)

# train test split
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'],
                                                    test_size=0.3, random_state=SEED, stratify=df['label'])

def train_val_test_split(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                np.ndarray,np.ndarray,np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=SEED,
                                                      stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df.drop('label', axis=1), df['label'], 0.3, 0.2)

