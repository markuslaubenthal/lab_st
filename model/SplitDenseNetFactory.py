from .DenseNetFactory import DenseNetFactory
from .TimeEmbeddingFactory import TimeEmbeddingFactory
import tensorflow as tf
from tensorflow.keras import layers
class SplitDenseNetFactory():
    def __init__(self):
        self.x = 0


    def Model(self, closeness_length=3, period_length=3, time_shape=(24+7+1,), time_embedding_method=0):
        dn_factory = DenseNetFactory()
        te_factory = TimeEmbeddingFactory()
        period_dependency_model, period_input = dn_factory.Model(prefix="period_dependency", input_shape=(100,100,period_length))
        closeness_dependency_model, closeness_input = dn_factory.Model(prefix="closeness_dependency", input_shape=(100,100,closeness_length))

        time_model, time_input = te_factory.Model(input_shape=time_shape)

        combined = layers.Add()([period_dependency_model, closeness_dependency_model])
        combined = self.TimeEmbeddingMethod(time_model, combined, method=time_embedding_method)

        combined = layers.Activation('sigmoid', name="output_sigmoid")(combined)

        combined = layers.Flatten()(combined)
        model = tf.keras.models.Model([period_input, closeness_input, time_input], combined)
        return model

    def TimeEmbeddingMethod(self, time_output, model, method = 0):
        # Simple Add
        if method == 0:
            return layers.Add()([model, time_output])
        # Hadamard scale
        if method == 1:
            time_model = HadamardLayer()(time_output)
            model = layers.Multiply()([model, time_model])
            return model
        # Scale from 0 - 2
        if method == 2:
            time_model = layers.Lambda(lambda x: x*2)(time_output)
            model = layers.Multiply()([model, time_model])
            return model
