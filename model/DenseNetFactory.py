import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from keras.optimizers import SGD, Adam
from .HadamardLayer import HadamardLayer
from keras.regularizers import l2
import numpy as np

import tensorflow.keras.backend as K


class DenseNetFactory():

    def __init__(self):
        self.concat_axis = 3
        self.eps = 1.1e-5
        self.growth_rate = 8
        self.initial_filters = 8
        self.num_conv_layer = 8
        self.weight_decay = 1e-4
        self.kernel_size = (3,3)
        self.kernel_regularizer = l2(self.weight_decay)
        self.use_bias = False

        
        self.kernel_regularizer = None
        self.use_bias = True

    def ConvLayer(self, x, name):
        x = layers.Conv2D(
            self.growth_rate,
            self.kernel_size,
            use_bias = self.use_bias,
            kernel_regularizer = self.kernel_regularizer,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            padding="same",
            name = name)(x)
        return x

    def DenseLayer(self, x, prefix, index):
        x = layers.BatchNormalization(
            epsilon=self.eps,
            axis=self.concat_axis,
            name = prefix + '_bn_dense' + str(index))(x)
        x = layers.Activation(
            'relu',
            name = prefix + '_relu_dense' + str(index))(x)
        x = self.ConvLayer(x, prefix + '_conv_dense' + str(index))
        return x


    def Model(self, prefix="standard", input_shape=None, input=None):
        if input is None:
            input = layers.Input(shape=input_shape, name = prefix + '_input')
        # x = self.ConvLayer(input, prefix + '_init_conv')
        x = input
        concatenationLayer = x
        for i in range(self.num_conv_layer):
            x = self.DenseLayer(x, prefix, i)
            if concatenationLayer is None:
                concatenationLayer = x
            else:
                concatenationLayer = layers.Concatenate(axis=self.concat_axis)([x, concatenationLayer])
                x = concatenationLayer
        model = layers.Conv1D(1,1, use_bias=False)(x) #Add activation Layer
        model = layers.Activation('linear')(model)
        # model = HadamardLayer(name = prefix + "_hadamard1")(model)
        return model, input
