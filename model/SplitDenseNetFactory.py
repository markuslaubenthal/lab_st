from .DenseNetFactory import DenseNetFactory
from .TimeEmbeddingFactory import TimeEmbeddingFactory
import tensorflow as tf
from tensorflow.keras import layers
class SplitDenseNetFactory():
    def __init__(self):
        self.x = 0


    def Model(self,
            closeness_length=3,
            period_length=3,
            time_shape=(24+7+1,),
            growth_rate = 8,
            initial_filters = 8,
            depth = 8):
        dn_factory = DenseNetFactory()
        dn_factory.growth_rate = growth_rate
        dn_factory.initial_filters = initial_filters
        dn_factory.num_conv_layer = depth
        te_factory = TimeEmbeddingFactory()
        period_dependency_model, period_input = dn_factory.Model(prefix="period_dependency", input_shape=(100,100,period_length))
        closeness_dependency_model, closeness_input = dn_factory.Model(prefix="closeness_dependency", input_shape=(100,100,closeness_length))

        time_model, time_input = te_factory.Model(input_shape=time_shape)

        combined = layers.Add()([period_dependency_model, closeness_dependency_model])
        combined = layers.Concatenate()([combined, time_model], concat_axis=3)
        combined, input = dn_factory.Model(prefix="Final_DenseNet", input=combined)
        combined = layers.Conv2D(1, (1,1))(combined)
        combined = layers.Activation('sigmoid', name="output_sigmoid")(combined)

        combined = layers.Flatten()(combined)
        model = tf.keras.models.Model([period_input, closeness_input, time_input], combined)
        return model
