import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import numpy as np
from GNN_layers import GNN_layer, MLP
from utils import sample_pairs

class GNN_BC(keras.Model):

    '''
    *Each block is essentially half of the network (one block takes the regular adjacency matrix A,
     the other takes A_transposed)
    *The blocks are composed by cells, each cell is a pair of (GNN_layer, MLP)
    '''

    def __init__(self, input_shape, num_nodes, params, real_BCs, name = "GNN_BC", **kwargs):
        super().__init__(name=name, **kwargs)
        self.real_BCs = real_BCs
        self.params = params
        self.num_nodes = num_nodes

        self.adj_block = self.make_block(params, input_shape, num_nodes)
        self.adj_t_block = self.make_block(params, input_shape, num_nodes)

    def call(self, x):
        flat_adj_matrix = x[0]
        flat_adj_matrix_t = x[1]

        adj_score = self.get_block_score(flat_adj_matrix, transposed = False)
        adj_t_score = self.get_block_score(flat_adj_matrix_t, transposed = True)

        final_score = tf.math.multiply(adj_score, adj_t_score)

        return final_score

    def make_block(self, params, input_shape, num_nodes):
        n_cells = params['n_messaging_cells']
        num_hidden = params['hidden_neurons_messaging_cells']

        block = []
        for i in range(n_cells):
            gnn_layer = GNN_layer(input_shape, num_hidden, params, i)

            mlp = MLP(params, input_shape= num_hidden, output_shape= num_nodes, index=i)

            cell = (gnn_layer, mlp)
            block.append(cell)

            input_shape = num_hidden

        return block

    def get_block_score(self, input, transposed):
        if transposed:
            block = self.adj_t_block
        else:
            block = self.adj_block

        mlp_scores = []
        for cell in block:
            input = cell[0](input)

            score = cell[1](input)
            mlp_scores.append(score)

        block_score = tf.math.add_n(mlp_scores)
        return block_score


    def loss_function(self, pred, real, y, margin):
        loss = tf.math.maximum(tf.constant(0, dtype=np.float32), tf.add(tf.math.multiply(-y, tf.subtract(pred,real)) , margin))

        return loss

    def train_step(self, data):
        x = data[0]
        with tf.GradientTape() as tape:
            scores = self.call(x)
            pairs_list = sample_pairs(self.num_nodes, self.params['pairs_sample_size'])

            total_loss=tf.constant(0, dtype=np.float32)
            for pair in pairs_list:
                real_difference = tf.constant(self.real_BCs[int(pair[0])] - self.real_BCs[int(pair[1])], dtype= np.float32)
                predicted_difference = tf.math.subtract( scores[0][int(pair[0])], scores[0][int(pair[1])] )

                y = 1 if tf.math.greater(predicted_difference, real_difference) else -1

                pair_loss = self.loss_function(predicted_difference,
                                               real_difference,
                                               y,
                                               self.params['margin'])
                total_loss = tf.math.add(total_loss, pair_loss)



        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": total_loss}