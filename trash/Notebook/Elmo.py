import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
import keras
import numpy as np

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = np.asarray(list(data['Category']))
x_train = np.asarray(list(data['title']))

x_test = np.asarray(list(test['title']))

# le = preprocessing.LabelEncoder()
# le.fit(y)

# def encode(le, labels):
#     enc = le.transform(labels)
#     return keras.utils.to_categorical(enc)

# def decode(le, one_hot):
#     dec = np.argmax(one_hot, axis=1)
#     return le.inverse_transform(dec)

# test = encode(le, [i for i in range(58)])

# untest = decode(le, test)

# x_enc = x
# y_enc = np.asarray(y)

# x_train, x_test, y_train, y_test = train_test_split(np.asarray(x_enc), np.asarray(y), test_size=0.33, random_state=42,stratify = y)


from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
from keras.losses import binary_crossentropy ,sparse_categorical_crossentropy

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(58, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

print('start training______________________')
# with tf.Session() as session:
#     K.set_session(session)
#     session.run(tf.global_variables_initializer())  
#     session.run(tf.tables_initializer())
#     history = model.fit(x_train, y_train, epochs=1, batch_size=256)
#     model.save_weights('./elmo-model.h5')
history = model.fit(x_train, y_train, epochs=5, batch_size=1024)
model.save_weights('./elmo-model.h5')

print('Start prediction____________')
# with tf.Session() as session:
#     K.set_session(session)
#     session.run(tf.global_variables_initializer())
#     session.run(tf.tables_initializer())
#     model.load_weights('./elmo-model.h5')  
#     predicts = model.predict(x_test, batch_size=256)
#     oof_pred = model.predict(x_train, batch_size=512)
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
