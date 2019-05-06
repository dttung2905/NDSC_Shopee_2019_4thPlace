from keras.initializers import glorot_uniform, he_uniform, he_normal
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

#-------------reproducable result ------------
seed_nb=14
import numpy as np
np.random.seed(seed_nb)
import tensorflow as tf
tf.set_random_seed(seed_nb)
from keras.initializers import glorot_uniform, he_uniform, he_normal
#-------------reproducable result ------------






# https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
def build_model(embedding_matrix, nb_words, embedding_size=300):
    inp = Input(shape=(max_length,))
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.35,seed=seed_nb)(x)
    x1 = Bidirectional(CuDNNLSTM(256,kernel_initializer=glorot_uniform(seed=seed_nb) ,return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(128,kernel_initializer=glorot_uniform(seed=seed_nb) ,return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(58, activation='softmax',kernel_initializer=he_uniform(seed=seed_nb))(conc)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
	print('running model.py')
