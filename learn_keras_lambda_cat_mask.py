import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import numpy as np

"""
本部分主要实验 keras 的 concat +mask 函数

理解参考 python 的 lambda 语法
https://realpython.com/python-lambda/
"""
# [B,CT]=>[B,CT,1]
x1_in = Input(shape=(None,), name='x1_in')
x_up_2 = Lambda(lambda x: x ** 2)(x1_in)

embedding = Embedding(1000, 10, name='char_emb')
x_up_2_emb = embedding(x_up_2)  # [4,3][4,3,10]

dense = Dense(5, use_bias=False, name='char_dense')  # 这里神经元数量对应输出的 dimension
x_up_2_emb_dense = dense(x_up_2_emb)  # [4,3,10]=[4,3,5]
cat_ = Concatenate()([x_up_2_emb, x_up_2_emb_dense])

x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x1_mask')(x1_in)

cat_mask_ = Lambda(lambda x: x[0] * x[1], name='x1')([cat_, x1_mask])  # [4,3,15]*[4,3,1] 这里应用了广播机制

fn = K.function([x1_in], [x1_mask, x_up_2, x_up_2_emb, x_up_2_emb_dense, cat_, cat_mask_])

arr = [[1, 2, 3],
       [1, 2, 0],
       [1, 0, 0],
       [0, 0, 0]]
arr = np.asarray(arr)  # batch_size=4, time_step=3
res = fn([arr])
for r in res[-2:]:
    print(r)
    print(r.shape)
