from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Code from https://github.com/zctzzy/traffic_prediction
class HadamardLayer(Layer):
    def __init__(self, **kwargs):
        super(HadamardLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:]) - 0.5
        self.W = K.variable(initial_weight_value)
        self._trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape
