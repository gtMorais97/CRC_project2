import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import numpy as np
from GNN_layers import GNN_layer, MLP
from utils import sample_pairs
from random import randint

class GNN_BC_2(keras.Model):

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

        self.block = self.make_block(params, input_shape, num_nodes)
        self.mlp = MLP(params, input_shape=params['hidden_neurons_messaging_cells'], output_shape=num_nodes)

    def call(self, x):
        flat_adj_matrix = x[0]
        flat_adj_matrix_t = x[1]

        adj_score = self.get_block_score(flat_adj_matrix)
        adj_t_score = self.get_block_score(flat_adj_matrix_t)

        final_score = tf.math.multiply(adj_score, adj_t_score)

        return final_score

    def make_block(self, params, input_shape, num_nodes):
        n_cells = params['n_messaging_cells']
        num_hidden = params['hidden_neurons_messaging_cells']

        block = []
        for i in range(n_cells):
            gnn_layer = GNN_layer(input_shape, num_hidden, params, i)
            block.append(gnn_layer)

        return block

    def get_block_score(self, input):
        mlp_scores = []
        first_block = True
        for gnn_layer in self.block:       
            if not first_block:
                z = gnn_layer(input)
                z = tf.multiply(z, aux)
                aux = z
            else:
                z = gnn_layer(input) #grey block
                aux = z

            score = self.mlp(z) #mlp
            mlp_scores.append(score)

            first_block = False

        block_score = tf.math.add_n(mlp_scores)
        return block_score


    def loss_function(self, node_i, node_j, y, margin):
        return tf.math.maximum(tf.constant(0, dtype=np.float32), tf.add(tf.math.multiply(-y, tf.subtract(node_i,node_j)) , margin))

    def train_step(self, data):
        x = data[0]
        real_BCs = self.real_BCs
        with tf.GradientTape() as tape:
            scores = self.call(x)
            #pairs_list = sample_pairs(self.num_nodes, self.params['pairs_sample_size'])
            ranking = np.argsort(np.negative(list(real_BCs.values())))
            total_loss=tf.constant(0, dtype=np.float32)
            for _ in range(self.params['pairs_sample_size']):
                rank_1 = randint(0, self.num_nodes-1)
                rank_2 = randint(0, self.num_nodes-1)
                #real_difference = tf.constant(self.real_BCs[int(pair[0])] - self.real_BCs[int(pair[1])], dtype= np.float32)
                #predicted_difference = tf.math.subtract( scores[0][int(pair[0])], scores[0][int(pair[1])] )

                #y = 1 if tf.math.greater(predicted_difference, real_difference) else -1
                #y = 1 if list(sorted_BCs.keys()).index(i) < list(sorted_BCs.keys()).index(j) else -1
                y = -1. if rank_1>rank_2 else 1.

                pair_loss = self.loss_function(scores[0][ranking[rank_1]],
                                               scores[0][ranking[rank_2]],
                                               y,
                                               self.params['margin'])
                total_loss = tf.math.add(total_loss, pair_loss)

                del rank_1
                del rank_2


        print('calculating grads...')
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print('finished')
        return {"loss": total_loss}