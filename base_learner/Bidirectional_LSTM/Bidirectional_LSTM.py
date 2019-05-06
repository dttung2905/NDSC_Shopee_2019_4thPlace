from __future__ import absolute_import, division


#-------------reproducable result ------------
seed_nb=14
import numpy as np
np.random.seed(seed_nb)
import tensorflow as tf
tf.set_random_seed(seed_nb)
from keras.initializers import glorot_uniform, he_uniform, he_normal
#-------------reproducable result ------------

import os
import time
import pandas as pd
import gensim
from tqdm import tqdm
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")
import gc
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import sys
from os.path import dirname
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
from utils.loadembed import load_fasttext, load_para, load_glove
import spacy
from utils.model import build_model
# https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
# hyperparameters
max_length = 55
embedding_size = 600
learning_rate = 0.001
batch_size = 512
num_epoch = 6

glove_path = '../input/popular-embedding/embeddings/glove.840B.300d/glove.840B.300d.txt'
para_path =  '../input/popular-embedding/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
fasttext_path = '../input/popular-embedding/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
train_path = '../input/ndsc-beginner/train.csv'
test_path = '../input/ndsc-beginner/test.csv'
#batch_size = 128


# import data and load dictionary to build embedding-------------------------------------------------------------
start_time = time.time()
print("Loading data ...")
train = pd.read_csv(train_path).fillna(' ')
test = pd.read_csv(test_path).fillna(' ')
train_text = train['title']
test_text = test['title']
text_list = pd.concat([train_text, test_text])
y = train['Category'].values
num_train_data = y.shape[0]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Spacy NLP ...")
nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
word_dict = {}
word_index = 1
lemma_dict = {}
docs = nlp.pipe(text_list, n_threads = 2)
word_sequences = []
for doc in tqdm(docs):
    word_seq = []
    for token in doc:
        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
            word_dict[token.text] = word_index
            word_index += 1
            lemma_dict[token.text] = token.lemma_
        if token.pos_ is not "PUNCT":
            word_seq.append(word_dict[token.text])
    word_sequences.append(word_seq)
del docs
gc.collect()
train_word_sequences = word_sequences[:num_train_data]
test_word_sequences = word_sequences[num_train_data:]
print("--- %s seconds ---" % (time.time() - start_time))


def batch_gen(train_df):
    n_batches = math.floor(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            batch_labels = np.array(train_df["Category"][i*batch_size:(i+1)*batch_size])
            yield text_arr, batch_labels

# padd sequences and load embedding --------------------------------------------------------
train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')
test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')
print(train_word_sequences[:1])
print(test_word_sequences[:1])
pred_prob = np.zeros((len(test_word_sequences),58), dtype=np.float32)

start_time = time.time()
print("Loading embedding matrix ...")
embedding_matrix_glove, nb_words = load_glove(word_dict, lemma_dict,glove_path)
embedding_matrix_fasttext, nb_words = load_fasttext(word_dict, lemma_dict,fasttext_path)
embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext), axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

# Trainning time ----------------------------------------------------------------------------
start_time = time.time()
print("Start training ...")
model = build_model(embedding_matrix, nb_words, embedding_size)
all_preds = []
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)

pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)
pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))

del model, embedding_matrix_fasttext, embedding_matrix
gc.collect()
K.clear_session()
print("--- %s seconds ---" % (time.time() - start_time))

#Training another model ---------------------------------------------------------------------
start_time = time.time()
print("Loading embedding matrix ...")
embedding_matrix_para, nb_words = load_para(word_dict, lemma_dict,para_path)
embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_para), axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Start training ...")
model = build_model(embedding_matrix, nb_words, embedding_size)
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)
pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)
pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
np.save('pred_prob1.npy', pred_prob)
print("--- %s seconds ---" % (time.time() - start_time))
submission = pd.read_csv('../input/ndsc-beginner/data_info_val_sample_submission.csv')
y_te = [np.argmax(pred) for pred in pred_prob]
submission['Category'] = y_te

submission.to_csv('submission.csv', index=False)
