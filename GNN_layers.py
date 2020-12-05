import tensorflow as tf
import keras
from keras import layers

class GNN_layer(keras.Model):

    def __init__(self, input_shape, output_shape, params, index ,name="GNN_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        str_i = str(index)

        self.input_layer = layers.Input(input_shape, name = 'GNN_input_layer_'+str_i)

        self.output_layer = layers.Dense(output_shape, activation=params['GNN_layer_activation'], name='output_'+str_i)

    def call(self, input):
        output = self.output_layer(input)
        return output


class MLP(keras.Model):

    def __init__(self, params, input_shape, output_shape, index ,name="GNN_first_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        str_i = str(index)

        self.input_layer = layers.Input(input_shape, name = 'inputMLP_'+str_i )

        self.hidden_layers = []
        activation = params['mlp_activation']
        for j, num_neurons in enumerate(params['mlp_hidden_layers']):
            layer = layers.Dense(num_neurons, activation= activation, name = 'mlp_'+str_i+'_layer_'+str(j))
            self.hidden_layers.append(layer)

        self.output_layer = layers.Dense(output_shape, activation=params['mlp_activation'])



    def call(self, input):
        for layer in self.hidden_layers:
            input = layer(input)

        output = self.output_layer(input)
        return output