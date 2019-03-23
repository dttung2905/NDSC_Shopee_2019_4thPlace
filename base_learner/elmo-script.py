import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout2D
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
import keras
import numpy as np

seed_nb = 100
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = np.asarray(list(data['Category']))
x_train = np.asarray(list(data['title']))

x_test = np.asarray(list(test['title']))



from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
from keras.losses import binary_crossentropy ,sparse_categorical_crossentropy
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.initializers import glorot_uniform, he_uniform, he_normal

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(512, activation='relu')(embedding)
dense = Dense(256, activation='relu')(dense)

pred = Dense(58, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print('start training______________________')


history = model.fit(x_train, y_train, epochs=5, batch_size=512)
model.save_weights('./elmo-model.h5')

print('Start prediction____________')

model.load_weights('./elmo-model.h5')
predicts = model.predict(x_test, batch_size=256,verbose = 1)
oof_pred = model.predict(x_train, batch_size=512,verbose = 1)
        
print('Done prediction')

np.save('oof_Elmo.np',oof_pred)
print('finish saving oof numpy array')

y_te = [np.argmax(preds) for preds in predicts]

submission = pd.read_csv("../input/data_info_val_sample_submission.csv")
np.save('prediction_array.np',predicts)
print('finish saving numpy array')
y_te = [np.argmax(preds) for preds in predicts]
submission['Category'] = y_te
submission.to_csv("submission.csv", index = False)
print('Done-----------')
