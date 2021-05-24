import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import numpy as np

"""
本部分主要实验 keras 的 lambda 函数

理解参考 python 的 lambda 语法
https://realpython.com/python-lambda/
"""
# [B,CT]=>[B,CT,1]
x1_in = Input(shape=(None,), name='x1_in')
x_up_2 = Lambda(lambda x: x ** 2)(x1_in)

embedding = Embedding(1000, 10, name='char_emb')
x_up_2_emb = embedding(x_up_2)

dense = Dense(5, use_bias=False, name='char_dense')  # 这里神经元数量对应输出的 dimension
x_up_2_emb_dense = dense(x_up_2_emb)  # [4,3,10]=[4,3,5]

x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x1_mask')(x1_in)
fn = K.function([x1_in], [x1_mask, x_up_2, x_up_2_emb, x_up_2_emb_dense])

arr = [[1, 2, 3],
       [1, 2, 0],
       [1, 0, 0],
       [0, 0, 0]]
arr = np.asarray(arr)  # batch_size=4, time_step=3
res = fn([arr])
for r in res:
    print(r)
    print(r.shape)
