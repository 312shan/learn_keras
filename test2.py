import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import numpy as np
import pickle

word_size = 256
num_features = 3
char_size = 128

data = pickle.load(open('data.pkl', 'rb'))
len(data)
id2char = pickle.load(open('id2char.pkl', 'rb'))


class MyBidirectional:
    """自己封装双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer):
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, inputs):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        x, mask = inputs
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def __call__(self, inputs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = Lambda(self.reverse_sequence)([x, mask])
        x_backward = self.backward_layer(x_backward)
        x_backward = Lambda(self.reverse_sequence)([x_backward, mask])
        x = Concatenate()([x_forward, x_backward])
        x = Lambda(lambda x: x[0] * x[1])([x, mask])
        return x


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

pres1 = Lambda(lambda x: K.expand_dims(x, 2), name='pres1')(pres1)  # [B,CT,1]
pres2 = Lambda(lambda x: K.expand_dims(x, 2), name='pres2')(pres2)  # [B,CT,1]
x1 = Concatenate()([x1, pres1, pres2])
x1_after_cat = x1

x1 = Lambda(lambda x: x[0] * x[1], name='x1')([x1, x1_mask])  # 进行 mask,mask 掉padding 的部分

x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True, name='lstm_1'))([x1, x1_mask])
x1_after_lstm1 = x1  # [B,CT,CH]

# [B,CT,CH] ,因为 padding='same' CT 不变，CH 不变因为filter_num == CH
h = Conv1D(char_size, 3, activation='relu', padding='same', name='h')(x1)

ps1 = Dense(1, activation='sigmoid')(h)  # [B,CT,1]
ps2 = Dense(1, activation='sigmoid')(h)  # [B,CT,1]

ps1_before_lambda, ps2_before_lambda = ps1, ps2

# 在 pres1 中为0的，预测值也必须为零，就是这个意思
ps1 = Lambda(lambda x: x[0] * x[1], name='ps1')([ps1, pres1])  # 这样一乘，相当于只从最大匹配的结果中筛选实体
ps2 = Lambda(lambda x: x[0] * x[1], name='ps2')([ps2, pres2])  # 这样一乘，相当于只从最大匹配的结果中筛选实体
ps1_after_lambda, ps2_after_lambda = ps1, ps2

s_model = Model([x1_in, x1v_in, pres1_in, pres2_in], [ps1, ps2])

x1 = Concatenate()([x1, y])
x1_after_cat_y = x1  # [B,CT,CH+4]
x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True, name='lstm_2'))([x1, x1_mask])
x1_after_lstm2 = x1  # [B,CT,CH]

# y1 是正解的实体所在的位置 y =[y1,y2,y3,y4], 先做乘法，去掉不可能的部分，然后得到概率权重， sum 这个操作不太懂，还需要进一步了解
ys = Lambda(lambda x: K.sum(x[0] * x[1][..., :1], 1) / K.sum(x[1][..., :1], 1), name='ys')([x1, y])  # [B,CH]

model = Model(inputs=[x1_in, x1v_in, pres1_in, pres2_in, y_in],
              outputs=[x1_after_add, x1_after_cat, x1_after_lstm1, h, ps1_before_lambda, ps2_before_lambda,
                       ps1_after_lambda, ps2_after_lambda, x1_after_cat_y, x1_after_lstm2, ys])

X1, X2, X1V, X2V, S1, S2, PRES1, PRES2, Y, T = data[0]
print(Y.shape)
res = model.predict([X1, X1V, PRES1, PRES2, Y])

keys = ['x1_after_add', 'x1_after_cat', 'x1_after_lstm1', 'h', 'ps1_before_lambda', 'ps2_before_lambda', 'ps1', 'ps2',
        'x1_after_cat_y', 'x1_after_lstm2', 'ys']
for k, i in zip(keys, res):
    print('{} : {}'.format(k, i.shape))
