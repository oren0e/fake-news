import pandas as pd
import numpy as np

from typing import Tuple

import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

SEED = 104

# read the data
df: pd.DataFrame = pd.read_pickle('./data/df.pkl')

# combine title and text
df['text'] = df['title'] + ' ' + df['text']

# remove unused variables
df.drop(['date', 'num_date', 'index', 'title'], axis=1, inplace=True)

# train test split
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'],
                                                    test_size=0.3, random_state=SEED, stratify=df['label'])

# write function to remove punctuation and
def text_process(text: str) -> str:
    no_punc = ''.join([word for word in text.rstrip() if word not in string.punctuation]).lower()
    word_tokens = nltk.word_tokenize(no_punc)
    # TODO: consider removing the stopwords filter depending on the results of the model
    no_stopwords = ''.join([word for word in word_tokens if word not in stopwords.words('english')])

    return no_stopwords

# find out how many words are in each text
a = X_train['text'].apply(lambda x: len(x.split(' ')))
a.describe()    # mean num_words =~ 400, we'll use half of that.
del a

max_features = 10000
maxlen = 200

# X_train['text'] = X_train['text'].apply(text_process)   # slow!

# get numbered representation of the corpus
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train['text'].values)
seqs_train = tokenizer.texts_to_sequences(X_train['text'].values)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_test['text'].values)
seqs_test = tokenizer.texts_to_sequences(X_test['text'].values)

# cut at maxlen words and pad with zeroes if necessary
seqs_train = sequence.pad_sequences(seqs_train, maxlen=maxlen)
seqs_test = sequence.pad_sequences(seqs_test, maxlen=maxlen)

