import graph_nets as gn
import sonnet as snt
import numpy as np
import tensorflow as tf

SMALL_GRAPH_1 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 1],
    "receivers": [1, 2],
}

SMALL_GRAPH_2 = {
    "globals": [-1.1, -1.2, -1.3],
    "nodes": [[-10.1, -10.2], [-20.1, -20.2], [-30.1, -30.2]],
    "edges": [[-101., -102., -103., -104.]],
    "senders": [1,],
    "receivers": [2,],
}

SMALL_GRAPH_3 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [1, 1],
    "receivers": [0, 2],
}

SMALL_GRAPH_4 = {
    "globals": [1.1, 1.2, 1.3],
    "nodes": [[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]],
    "edges": [[101., 102., 103., 104.], [201., 202., 203., 204.]],
    "senders": [0, 2],
    "receivers": [1, 1],
}

from graph_nets import utils_tf

def get_graphs():
    input_graph = utils_tf.data_dicts_to_graphs_tuple(
        [SMALL_GRAPH_1, SMALL_GRAPH_2, SMALL_GRAPH_3, SMALL_GRAPH_4])
    print(input_graph)
    return input_graph

# Provide your own functions to generate graph-structured data.



input_graphs = get_graphs()

# Create the graph network.
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

# Pass the input graphs to the graph network, and return the output graphs.
output_graphs = graph_net_module(input_graphs)

with tf.Session() as sess:
    print("training!!\n")
    sess.run(tf.global_variables_initializer())
    outres=sess.run(output_graphs)
    print(outres)