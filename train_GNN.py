import networkx as nx
from networkx.convert_matrix import to_numpy_array
import numpy as np
from utils import get_graph, get_bc_info
from GNN_BC import GNN_BC
import tensorflow as tf
import keras

params = {
    'num_epochs': 20,
    'learning_rate' : 0.0001,
    'n_messaging_cells': 3,
    'hidden_neurons_messaging_cells': 20,
    'mlp_hidden_layers': [20,15],
    'GNN_layer_activation': tf.nn.relu,
    'mlp_activation': tf.nn.relu,
    'pairs_sample_size': 50,
    'margin': 1
}

files=["email.txt"]

for file in files:
    file_tokens = file.split(".")
    file_name = file_tokens[0]
    file_type = file_tokens[1]

    g = get_graph("graphs/"+file, file_type)
    info_dict = get_bc_info(g)
    adj_matrix = to_numpy_array(g)
    adj_matrix_t = np.transpose(adj_matrix)
    print(adj_matrix.shape)

    flat_adj_matrix = np.ndarray.flatten(adj_matrix)
    flat_adj_matrix_t = np.ndarray.flatten(adj_matrix_t)

    real_BCs = np.asarray(list(info_dict['BC_dict'].values()))

    optimizer = keras.optimizers.Adam(params['learning_rate'])
    gnn = GNN_BC(flat_adj_matrix.shape[0], adj_matrix.shape[0], params, real_BCs)
    gnn.compile(optimizer)

    gnn.fit(flat_adj_matrix, real_BCs, epochs= params['num_epochs'], verbose=True)


