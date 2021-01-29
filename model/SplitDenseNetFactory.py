from .DenseNetFactory import DenseNetFactory
from .TimeEmbeddingFactory import TimeEmbeddingFactory
import tensorflow as tf
from tensorflow.keras import layers
from .HadamardLayer import HadamardLayer
import tensorflow.keras.backend as K

class SplitDenseNetFactory():
    def __init__(self):
        self.x = 0


    def Model(self,
            input_length=(63,),
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



        grid_inputs = []
        models = []
        inputs = []
        for i, input in enumerate(input_length):
            _model, _input = dn_factory.Model(prefix=str(i) + "_dependency", input_shape=(100,100,input))
            inputs.append(_input)
            grid_inputs.append(_input)
            models.append(_model)
        # period_dependency_model, period_input = dn_factory.Model(prefix="period_dependency", input_shape=(100,100,period_length))
        # closeness_dependency_model, closeness_input = dn_factory.Model(prefix="closeness_dependency", input_shape=(100,100,closeness_length))
        # inputs = grid_inputs.copy()

        t_m1_input = None
        if t_minus_one:
            t_m1_input = tf.keras.Input(name="t_minus_1_input", shape=(100,100,1))
            inputs.append(t_m1_input)

        if(len(models) > 1):
            combined = layers.Add()(models)
        else:
            combined = models[0]

        if time_embedding_method is not None:
            time_model, time_input = te_factory.Model(input_shape=time_shape)
            combined = self.TimeEmbeddingMethod(time_model, combined, method=time_embedding_method, t_m1_input=t_m1_input)
            inputs.append(time_input)


        combined = layers.Flatten()(combined)
        input_concatenation = layers.Concatenate(axis=3)(grid_inputs)
        input_concatenation = layers.Reshape((100*100, -1))(input_concatenation)
        input_concatenation = layers.Permute((2,1))(input_concatenation)
        input_concatenation = layers.Conv1D(32,1)(input_concatenation)
        input_concatenation = layers.Permute((2,1))(input_concatenation)

        combined = layers.Attention()([input_concatenation, combined, input_concatenation])

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
            model = layers.Lambda(lambda x: K.sum(x, axis=3))(model)
            # model = layers.Conv1D(1,1, use_bias=False)(model)
            return model
