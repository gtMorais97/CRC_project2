import tensorflow as tf
import keras
from keras import layers
import numpy as np
from random import randint

class NeuralNet(keras.Model):

    def __init__(self, input_shape, output_shape, params, name="NeuralNet", **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params

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

    def loss_function(self, node_i, node_j, y, margin):
        return tf.math.maximum(tf.constant(0, dtype=np.float32), tf.add(tf.math.multiply(-y, tf.subtract(node_i,node_j)) , margin))

    def train_step(self, data):

        x = data[0][0]
        print('x shape:', x.shape)
        real_BCs = self.real_BCs
        with tf.GradientTape() as tape:
            scores = self.call(x)
            print(scores.shape)
            ranking = np.argsort(np.negative(list(real_BCs.values())))
            total_loss=tf.constant(0, dtype=np.float32)
            for _ in range(self.params['pairs_sample_size']):
                rank_1 = randint(0, len(real_BCs)-1)
                rank_2 = randint(0, len(real_BCs)-1)

                y = tf.cast(tf.math.sign(-1*(rank_1-rank_2)), np.float32)

                pair_loss = self.loss_function(scores[ranking[rank_1]],
                                               scores[ranking[rank_2]],
                                               y,
                                               self.params['margin'])
                total_loss = tf.math.add(total_loss, pair_loss)

                del rank_1
                del rank_2

        del ranking
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}
