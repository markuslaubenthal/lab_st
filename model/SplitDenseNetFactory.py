from .DenseNetFactory import DenseNetFactory
from .TimeEmbeddingFactory import TimeEmbeddingFactory
import tensorflow as tf
from tensorflow.keras import layers
from .HadamardLayer import HadamardLayer


class SplitDenseNetFactory():
    def __init__(self):
        self.x = 0


    def Model(self,
            closeness_length=3,
            period_length=3,
            time_shape=(24+7+1,),
            growth_rate = 8,
            initial_filters = 8,
            depth = 8,
            time_embedding_method=None,
            t_minus_one=False):

        dn_factory = DenseNetFactory()
        dn_factory.growth_rate = growth_rate
        dn_factory.initial_filters = initial_filters
        dn_factory.num_conv_layer = depth
        te_factory = TimeEmbeddingFactory()

        # if time_embedding_method == "in_front":
        #     time_model, time_input = te_factory.Model(input_shape=time_shape)
        #     time_embedding = self.TimeEmbeddingMethod(time_model, combined, method=time_embedding_method)


        period_dependency_model, period_input = dn_factory.Model(prefix="period_dependency", input_shape=(100,100,period_length))
        closeness_dependency_model, closeness_input = dn_factory.Model(prefix="closeness_dependency", input_shape=(100,100,closeness_length))
        inputs = [period_input, closeness_input]

        t_m1_input = None
        if t_minus_one:
            t_m1_input = tf.keras.Input(name="t_minus_1_input", shape=(100,100,1))
            inputs.append(t_m1_input)

        combined = layers.Add()([period_dependency_model, closeness_dependency_model])

        if time_embedding_method is not None:
            time_model, time_input = te_factory.Model(input_shape=time_shape)
            combined = self.TimeEmbeddingMethod(time_model, combined, method=time_embedding_method, t_m1_input=t_m1_input)
            inputs.append(time_input)

        combined = layers.Activation('sigmoid', name="output_sigmoid")(combined)
        combined = layers.Flatten()(combined)

        model = tf.keras.models.Model(inputs, combined)
        return model

    def TimeEmbeddingMethod(self, time_output, model, method = "add", t_m1_input=None):
        # Simple Add
        if method == "add":
            return layers.Add()([model, time_output])
        # Hadamard scale
        if method == "weighted multiply":
            time_model = HadamardLayer()(time_output)
            model = layers.Multiply()([model, time_model])
            return model
        # Scale from 0 - 2
        if method == "scale_0_2":
            time_model = layers.Lambda(lambda x: x*2)(time_output)
            model = layers.Multiply()([model, time_model])
            return model
        if method == 't_minus_one':
            if t_m1_input is None:
                print("t_m1_input can not be None")
            time_model = layers.Multiply()([time_output, t_m1_input])
            model = layers.Concatenate(axis=3)([time_model, model])
            model = HadamardLayer()(model)
            model = layers.Conv1D(1,1, use_bias=False)(model)
            return model
