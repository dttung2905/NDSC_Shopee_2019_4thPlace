## Settings:
# some config values
max_features = 75825 #90000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

import os
import time
import tensorflow as tf
import numpy as np # linear algebra
import random
import os
os.environ['PYTHONHASHSEED'] = '11'
np.random.seed(22)
random.seed(33)
tf.set_random_seed(44)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import Callback
from keras.models import clone_model
import keras.backend as K
from utils.loadembed import load_glove, load_wiki, load_paragram,load_googlenews
from utils.ema import ExponentialMovingAverage

t0 = time.time()
train_path = "../input/ndsc-beginner/train.csv"
test_path = "../input/ndsc-beginner/train.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)

## fill up the missing values
train_X = train_df["title"].fillna("_na_").values
# val_X = val_df["title"].fillna("_na_").values
test_X = test_df["title"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'’“”')
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
# val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
trunc = 'pre'
train_X = pad_sequences(train_X, maxlen=maxlen, truncating=trunc)
# val_X = pad_sequences(val_X, maxlen=maxlen, truncating=trunc)
test_X = pad_sequences(test_X, maxlen=maxlen, truncating=trunc)

## Get the target values
train_y = train_df['Category'].values


#get the embedding
path_glove = '../input/popular-embedding/embeddings/glove.840B.300d/glove.840B.300d.txt'
path_wiki = '../input/popular-embedding/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
path_paragram = '../input/popular-embedding/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
path_w2v = '../input/popular-embedding/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embedding_matrix_1 =load_glove(path_glove,max_features)
embedding_matrix_2 =load_wiki(path_wiki, max_features)
embedding_matrix_3 =load_paragram(path_paragram, max_features)
embedding_matrix_4 =load_googlenews(path_w2v, max_features)

embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4), axis=1)
del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4
gc.collect()
np.shape(embedding_matrix)

print(f'Done preprocessing {time.time() - t0:.1f}s')


embed_ids = [list(range(300)), list(range(300, 600)),
             list(range(600, 900)), list(range(900, 1200))]
embed_ids_dict = {1: [embed_ids[0], embed_ids[1], embed_ids[2], embed_ids[3]],
                  2: [embed_ids[0] + embed_ids[1],
                      embed_ids[0] + embed_ids[2],
                      embed_ids[0] + embed_ids[3],
                      embed_ids[1] + embed_ids[2],
                      embed_ids[1] + embed_ids[3],
                      embed_ids[2] + embed_ids[3]],
                  3: [embed_ids[0] + embed_ids[1] + embed_ids[2],
                      embed_ids[0] + embed_ids[1] + embed_ids[3],
                      embed_ids[0] + embed_ids[2] + embed_ids[3],
                      embed_ids[1] + embed_ids[2] + embed_ids[3]],
                  4: [embed_ids[0] + embed_ids[1] + embed_ids[2] + embed_ids[3]]}
embed_ids_lst = embed_ids_dict[2]
embed_size = 600

rnn = CuDNNGRU
embed_trainable = False

n_models = 6
epochs = 7
batch_size = 512
dense1_dim = rnn_dim = 128
dense2_dim = 2 * rnn_dim

ema_n = int(len(train_y) / batch_size / 10)
decay = 0.9
scores = []

oof_pred = np.zeros((len(train_X),58))
# pred_avg = np.zeros((len(val_y), 58))
pred_test_avg = np.zeros((test_df.shape[0], 58))
for i in range(n_models):
    t1 = time.time()
    seed = 101 + 11 * i
    cols_in_use = embed_ids_lst[i % len(embed_ids_lst)]
    model = create_rnn_model(rnn, maxlen, embedding_matrix[:, cols_in_use],
                             max_features, embed_size,
                             rnn_dim=rnn_dim,
                             dense1_dim=dense1_dim,
                             dense2_dim=dense2_dim,
                             embed_trainable=embed_trainable,
                             seed=seed)
    ema = ExponentialMovingAverage(model, decay=decay, mode='batch', n=ema_n)
    model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs,
              callbacks=[ema], verbose=1)
    m = ema.ema_model
    t_per_epoch = (time.time() - t1) / epochs
#     pred = m.predict([val_X])
    train_pred = m.predict([train_X])
#     print(pred.shape)
    oof_pred += train_pred
    pred_test = m.predict([test_X])
    pred_test_avg += pred_test
#     f1_one, thresh_one = f1_best(val_y, pred)
#     f1_avg, thresh_avg = f1_best(val_y, pred_avg / (i + 1))
#     nll_one = metrics.log_loss(val_y, pred)
#     nll_avg = metrics.log_loss(val_y, pred_avg / (i + 1))
#     auc_one = metrics.roc_auc_score(val_y, pred)
#     auc_avg = metrics.roc_auc_score(val_y, pred_avg)
    print(f'  n_model:{i + 1} epoch:{epochs} ' +
          f'Time:{time.time() - t1:.1f}s  {t_per_epoch:.1f}s/epoch')

