#Inspired by the script
#https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext/comments
#Version 5: dont have call back

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

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


train = pd.read_csv("../input/ndsc-beginner/train.csv")
test = pd.read_csv("../input/ndsc-beginner/test.csv")
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

embed_size = 300
max_features = 75720
max_len = 220

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

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences = True))(x1)
    x = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

    y = Bidirectional(LSTM(units, return_sequences = True))(x1)
    y = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)

    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)

    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)


    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    x = Dense(58, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, Y_train, batch_size = 256, epochs = 4,
                        verbose = 1
                        )
    model_json = model.to_json()


    with open("model_num.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model_num.h5")
    return model


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
