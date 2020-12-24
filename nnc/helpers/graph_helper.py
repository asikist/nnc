import json
import os

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import pandas as pd
from networkx.algorithms.matching import max_weight_matching

"""
Some basic graph utilities.
"""


def save_graph(path, G, pos):
    """
    Saves a networkx graph in a json file and its nodal layoput for plotting in npz to be loaded
    later with `nnc.helpers.graph_helpers.load_graph) method.
    :param path: The path to the file
    :param G: The `networkx.Graph` object
    :param pos: The position dictionary with format `node_id : [x, y]` denoting note index and
    layout coordinates.
    :return: None
    """
    pos = {x: tuple(y.tolist()) for x, y in pos.items()}
    nx.set_node_attributes(G, values=pos, name='pos')
    graph_json = json_graph.node_link_data(G)
    graph_path = os.path.join(path, 'graph.json')
    os.makedirs(path, exist_ok=True)
    with open(graph_path, "w+") as f1:
        json.dump(graph_json, f1)
    pos_path = os.path.join(path, 'pos.npz')
    np.savez(pos_path, pos)


def load_graph(path):
    """
    Loads a graph and its layout positons from given folder.
    :param path: The path where the folder containign the graph saved with
    `nnc.helpers.graph_helpers.save_graph) method.
    :return: a tuple with the graph object and the corresponding lauyout dictionary.
    """
    graph_path = os.path.join(path, 'graph.json')
    with open(graph_path, "r") as f1:
        graph_json = json.load(f1)
        G = json_graph.node_link_graph(graph_json)
        pos = nx.function.get_node_attributes(G, name='pos')
    return G, pos


def nx_maximum_matching(G):
    """
    The maximum method used to determine drivers nodes for CT-LTI systems according to:
    Liu, Y. Y., Slotine, J. J., & Barabási, A. L. (2011). Controllability of complex networks. nature, 473(7346), 167-173.
    :param G: The `networkx.Graph` objext
    :return: The sorted list of node indices that can be used as drivers.
    """
    df = pd.DataFrame(list(max_weight_matching(G)))
    drivers = set(G.nodes) - set(df[1].tolist())
    return sorted(list(drivers))


def square_lattice_position(n_nodes):
    """
    A method that creates a layout dictionary for square lattices in a grid format, with edges
    parallel to the cartesian axes.
    :param n_nodes: Number of nodes in lattice
    :return:  The position dictionary with format `node_id : [x, y]` denoting note index and
    layout coordinates.
    """
    side_size = int(np.sqrt(n_nodes))
    pos = {i: np.array([int(i // side_size), i % side_size]) for i in range(n_nodes)}
    return pos
