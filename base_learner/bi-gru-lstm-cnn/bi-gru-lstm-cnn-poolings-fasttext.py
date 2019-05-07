#Inspired by the script
#https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext/comments

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from utils.model import build_model
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

#general setting for model
embed_size = 300
max_features = 75720
max_len = 220
train_path = "../input/ndsc-beginner/train.csv"
test_path ="../input/ndsc-beginner/test.csv"
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

list_classes = [i for i in range(58)]
y = train['Category'].values
train["title"].fillna("no comment")
test["title"].fillna("no comment")

X_train = train
Y_train = y
del train
del y

raw_text_train = X_train["title"].str.lower()
raw_text_test = test["title"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text_train)
X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

X_train = pad_sequences(X_train.comment_seq, maxlen = max_len)
test = pad_sequences(test.comment_seq, maxlen = max_len)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)


model = build_model(lr = 1e-3, lr_d = 0, units = 112, dr = 0.2)
pred = model.predict(test, batch_size = 1024, verbose = 1)
oof_pred = model.predict(X_train,batch_size = 1024, verbose = 1)

#Saving oof_pred
np.save('oof_Bi-GRU-LSTM-CNN-Poolings-Fasttext.np',oof_pred)
print('finish saving oof numpy array')
submission = pd.read_csv("../input/ndsc-beginner/data_info_val_sample_submission.csv")

np.save('prediction_array.np',pred)
print('finish saving test numpy array')
y_te = [np.argmax(preds) for preds in pred]
submission['Category'] = y_te
submission.to_csv("submission.csv", index = False)
print("[{}] Completed!".format(time.time() - start_time))
