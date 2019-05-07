from keras.layers import Dense, Input, Embedding, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import Callback
from keras.models import clone_model
import keras.backend as K

def create_rnn_model(rnn, maxlen, embedding, max_features, embed_size,
                     rnn_dim=64, dense1_dim=100, dense2_dim=50,
                     embed_trainable=False, seed=123):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding],
                  trainable=embed_trainable)(inp)
    x = Dense(dense1_dim, activation='relu',
              kernel_initializer=glorot_uniform(seed=seed))(x)
    x = Bidirectional(rnn(rnn_dim, return_sequences=True,
                          kernel_initializer=glorot_uniform(seed=seed)))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(dense2_dim, activation='relu',
              kernel_initializer=glorot_uniform(seed=seed))(x)
    x = Dense(58, activation='softmax',
              kernel_initializer=glorot_uniform(seed=seed))(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print('running model.py')
