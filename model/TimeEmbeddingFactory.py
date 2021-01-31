# Author: Markus Laubenthal

import tensorflow as tf
from tensorflow.keras import layers


class TimeEmbeddingFactory():
    def __init__(self):
        x = 0

    def Model(self, input_shape):
        input = layers.Input(shape=input_shape, name = 'time_input')
        model = layers.Dense(32, activation='tanh', name = 'time_1')(input)
        model = layers.Dense(100*100, activation='tanh', name = 'time_2')(model)
        model = layers.Reshape((100,100,1))(model)
        return model, input
