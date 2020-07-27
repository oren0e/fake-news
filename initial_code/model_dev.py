import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models, layers
from tensorflow.keras.metrics import Precision, Recall, AUC

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
# def text_process(text: str) -> str:
#     no_punc = ''.join([word for word in text.rstrip() if word not in string.punctuation]).lower()
#     word_tokens = nltk.word_tokenize(no_punc)
#     # TODO: consider removing the stopwords filter depending on the results of the model
#     no_stopwords = ''.join([word for word in word_tokens if word not in stopwords.words('english')])
#
#     return no_stopwords
#
# # find out how many words are in each text
# a = X_train['text'].apply(lambda x: len(x.split(' ')))
# a.describe()    # mean num_words =~ 400, we'll use half of that.
# del a

max_features = 10000
maxlen = 200
emb_dim = 100

# X_train['text'] = X_train['text'].apply(text_process)   # slow!

# get numbered representation of the corpus
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train['text'].values)
seqs_train = tokenizer.texts_to_sequences(X_train['text'].values)
print(f'Found {len(tokenizer.word_index)} unique tokens in train')

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_test['text'].values)
seqs_test = tokenizer.texts_to_sequences(X_test['text'].values)
print(f'Found {len(tokenizer.word_index)} unique tokens in test')

# cut at maxlen words and pad with zeroes if necessary
seqs_train = sequence.pad_sequences(seqs_train, maxlen=maxlen)
seqs_test = sequence.pad_sequences(seqs_test, maxlen=maxlen)
print('Shape of train data tensor: ', seqs_train.shape)
print('Shape of test data tensor: ', seqs_test.shape)
print('Shape of train label tensor: ', y_train.shape)
print('Shape of test label tensor: ', y_test.shape)

# spit to train and val sets
seqs_x_train, seqs_x_val, seqs_y_train, seqs_y_val = train_test_split(seqs_train, y_train,
                                                    test_size=0.2, random_state=SEED, stratify=y_train)

# Dense network as a quick starter
input_tensor = layers.Input((maxlen,))
kmodel = layers.Embedding(max_features, emb_dim)(input_tensor)
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dense(64, activation='relu')(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', Precision(), Recall(), AUC()])
history = model.fit(seqs_x_train, seqs_y_train, epochs=30, batch_size=32,
                    validation_data=(seqs_x_val, seqs_y_val))

# plot performance
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
auc = history.history['auc']
val_auc = history.history['val_auc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'ro', label='Training AUC')
plt.plot(epochs, val_acc, 'r', label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('Training and Validation AUC')
plt.legend()
plt.show()

model.evaluate(seqs_test, y_test)