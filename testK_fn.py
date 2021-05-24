import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import numpy as np
import pickle

num_classes = 3


class Normal(Layer):
    def __init__(self, **kwargs):
        super(Normal, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        self.built = True

    def call(self, x):
        self.x_normalized = K.l2_normalize(x, -1)
        return self.x_normalized * self.kernel


x_in = Input(shape=(784,))
x = x_in

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
normal = Normal()
x = normal(x)
x = Dense(num_classes, activation='softmax')(x)

fn = K.function([x_in], [normal.x_normalized])
res = fn([np.random.randn(64, 784)])
print(res[0].shape)


# https://github.com/keras-team/keras/issues/13244