import tensorflow as tf
import keras
from keras import layers

class GNN_layer_init(keras.Model):

    def __init__(self, input_shape, output_shape, params, index ,name="GNN_layer_init", **kwargs):
        super().__init__(name=name, **kwargs)
        str_i = str(index)

        self.input_layer = layers.Input(input_shape, name = 'GNN_input_layer_'+str_i)

        self.dense_layer = layers.Dense(input_shape, activation=params['GNN_layer_activation'])

    def call(self, input):
        x = self.output_layer(input)
        output = tf.math.multiply(x, input)
        return output


class GNN_layer(keras.Model):

    def __init__(self, input_shape, output_shape, params, index ,name="GNN_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        str_i = str(index)

        self.input_layer = layers.Input(input_shape, name = 'GNN_input_layer_'+str_i)

        self.output_layer = layers.Dense(output_shape, activation=params['GNN_layer_activation'],kernel_regularizer="l1", name='output_'+str_i)

    def call(self, input):
        output = self.output_layer(input)
        return output


class MLP(keras.Model):

    def __init__(self, params, input_shape, output_shape ,name="MLP", **kwargs):
        super().__init__(name=name, **kwargs)

        self.input_layer = layers.Input(input_shape, name = 'inputMLP' )

        self.hidden_layers = []
        activation = params['mlp_activation']
        for j, num_neurons in enumerate(params['mlp_hidden_layers']):
            layer = layers.Dense(num_neurons, activation= activation, kernel_regularizer="l1", name = 'mlp_layer_'+str(j))
            self.hidden_layers.append(layer)

        self.output_layer = layers.Dense(output_shape, kernel_regularizer = "l1")


    def call(self, input):
        for layer in self.hidden_layers:
            input = layer(input)

        output = self.output_layer(input)
        return output