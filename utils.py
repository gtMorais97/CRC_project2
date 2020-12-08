import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality
import operator
import time
from random import randint
import numpy as np


def get_bc_info(g, k_top= 5):
    start_time = time.time()
    BC_dict = betweenness_centrality(g)
    total_time = round(time.time() - start_time, 2)

    # list of pairs (node, bc_value)
    max_BCs = list(sorted(BC_dict.items(), key=operator.itemgetter(1), reverse=True)[:k_top])

    total_BC = 0
    for bc in BC_dict.values():
        total_BC += bc
    avg_BC = total_BC / len(BC_dict)

    max_total = 0
    for bc in max_BCs:
        max_total += bc[1]
    avg_max_BCs = max_total / len(max_BCs)

    return {"BC_dict": BC_dict,
            "avg_BC": avg_BC,
            "max_BCs": max_BCs,
            "avg_max_BCs": avg_max_BCs,
            "time": total_time}


def get_graph(file, file_type="txt"):
    if file_type == "txt":
        sep = None
    elif file_type == "csv":
        sep = ','

    G = nx.DiGraph()

    with open(file, "r", encoding="utf-8") as graph:
        lines = graph.readlines()

        for line in lines:
            edge = [int(i) for i in line.split(sep)]
            G.add_edge(edge[0], edge[1])

    return G

def sample_pairs(num_nodes, sample_size):
    pairs = []
    for i in range(sample_size):
        node1 = randint(0, num_nodes-1)
        node2 = randint(0, num_nodes-1)

        pairs.append((node1,node2))

    return np.asarray(pairs, dtype=np.float32)

def clean_adj_matrix(adj_matrix, real_BCs):
    aux = np.copy(adj_matrix)
    for row_number, row in enumerate(aux):
        if(real_BCs[row_number] == 0):        
            aux[row_number] = 0
     
    return aux