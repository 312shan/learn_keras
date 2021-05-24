import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
# from el import MyBidirectional
import numpy as np
import pickle

word_size = 256
num_features = 3
char_size = 128

data = pickle.load(open('data.pkl', 'rb'))
len(data)
id2char = pickle.load(open('id2char.pkl', 'rb'))

x1_in = Input(shape=(None,), name='x1_in')
x2_in = Input(shape=(None,), name='x2_in')
x1v_in = Input(shape=(None, word_size), name='x1v_in')
x2v_in = Input(shape=(None, word_size), name='x2v_in')
s1_in = Input(shape=(None,), name='s1_in')
s2_in = Input(shape=(None,), name='s2_in')
pres1_in = Input(shape=(None,), name='pres1_in')
pres2_in = Input(shape=(None,), name='pres2_in')
y_in = Input(shape=(None, 1 + num_features), name='y_in')
t_in = Input(shape=(1,), name='t_in')

x1, x2, x1v, x2v, s1, s2, pres1, pres2, y, t = (
    x1_in, x2_in, x1v_in, x2v_in, s1_in, s2_in, pres1_in, pres2_in, y_in, t_in
)

# [B,CT]=>[B,CT,1]
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x1_mask')(x1)
# [B,WT]=>[B,WT,1]
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x2_mask')(x2)

embedding = Embedding(len(id2char) + 2, char_size, name='char_emb')
dense = Dense(char_size, use_bias=False, name='char_dense')

x1 = embedding(x1)  # [B,CT]=>[B,CT,C_H]
x1v = dense(x1v)  # [B,WT,W_H]=>[B,WT,C_H] , 这里有一些预处理上的做法，导致这里 WT == CT , 所以下面才能 Add
x1 = Add()([x1, x1v])  # query 的 char_level + word_level
x1_after_add = x1
x1 = Dropout(0.2)(x1)  # [B,CT,C_H]

pres1 = Lambda(lambda x: K.expand_dims(x, 2), name='pres1')(pres1)
pres2 = Lambda(lambda x: K.expand_dims(x, 2), name='pres2')(pres2)
x1 = Concatenate()([x1, pres1, pres2])
x1_after_cat = x1

model = Model(inputs=[x1_in, x1v_in, pres1_in, pres2_in], outputs=[x1_after_add, x1_after_cat])

X1, X2, X1V, X2V, S1, S2, PRES1, PRES2, Y, T = data[0]
res = model.predict([X1, X1V, PRES1, PRES2])
for i in res:
    print(i.shape)
