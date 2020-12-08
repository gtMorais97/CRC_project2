import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import numpy as np
from GNN_layers import GNN_layer, MLP
from utils import sample_pairs
from random import randint

class GNN_BC(keras.Model):

    '''
    *Each block is essentially half of the network (one block takes the regular adjacency matrix A,
     the other takes A_transposed)
    *The blocks are composed by cells, each cell is a pair of (GNN_layer, MLP)
    '''

    def __init__(self, input_shape, output_shape, params, name = "GNN_BC", **kwargs):
        super().__init__(name=name, **kwargs)
        self.real_BCs = None
        self.params = params
        self.input_layer = layers.Input(input_shape) 
        self.adj_block = self.make_block(params, input_shape, output_shape)
        self.adj_t_block = self.make_block(params, input_shape, output_shape)

    def call(self, x):
        adj_matrix = x[0]
        adj_matrix_t = x[1]

        adj_score = self.get_block_score(adj_matrix, transposed = False)
        adj_t_score = self.get_block_score(adj_matrix_t, transposed = True)

        final_score = tf.math.multiply(adj_score, adj_t_score)

        return final_score

    def make_block(self, params, input_shape, output_shape):
        n_cells = params['n_messaging_cells']
        num_hidden = params['hidden_neurons_messaging_cells']

        block = []
        for i in range(n_cells):
            gnn_layer = GNN_layer(input_shape, num_hidden, params, i)

            mlp = MLP(params, num_hidden, output_shape)

            cell = (gnn_layer, mlp)
            block.append(cell)

        return block

    def get_block_score(self, input, transposed):
        if transposed:
            block = self.adj_t_block
        else:
            block = self.adj_block

        mlp_scores = []
        first_block = True
        for cell in block:       
            if not first_block:
                z = cell[0](input)
                z = tf.multiply(z, aux)
                aux = z
            else:
                z = cell[0](input) #grey block
                aux = z
            
            score = cell[1](z) #mlp
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
        print('calculating grads...')
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print('finished')
        return {"loss": total_loss}