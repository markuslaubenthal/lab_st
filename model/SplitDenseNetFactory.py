from .DenseNetFactory import DenseNetFactory
import tensorflow as tf
from tensorflow.keras import layers
class SplitDenseNetFactory():
    def __init__(self):
        self.x = 0


    def Model(self, closeness_length, period_length):
        dn_factory = DenseNetFactory()
        period_dependency_model, period_input = dn_factory.Model(prefix="period_dependency", input_shape=(100,100,period_length))
        closeness_dependency_model, closeness_input = dn_factory.Model(prefix="closeness_dependency", input_shape=(100,100,closeness_length))

        combined = layers.Add()([period_dependency_model, closeness_dependency_model])
        combined = layers.Activation('sigmoid', name="output_sigmoid")(combined)

        combined = layers.Flatten()(combined)
        model = tf.keras.models.Model([period_input, closeness_input], combined)
