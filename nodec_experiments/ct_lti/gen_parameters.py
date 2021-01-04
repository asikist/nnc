import random

from nnc.helpers.data_generators import random_seed, numpy_seed, nx_seed
from nnc.helpers.data_generators import \
    cca_state_generator
from nnc.helpers.torch_utils.graphs import maximum_matching_drivers, adjacency_tensor
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch


def default_graph_generator(n_nodes: int, graph_topology: str):
    """
    Generates a networkx graph object based on the given parameters.
    :param n_nodes: the number of nodes in graph
    :param graph_topology: The graph topology which can be a string value from:
    `['lattice', 'ba', 'tree']`, used to create a square-lattice,
    Barabasi-Albert and  random-tree graphs respectively.
    :return: The `networkx.Graph` object containing the graph.
    """
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    if graph_topology == "lattice":
        side_size = int(n_nodes**(1/2))
        g = nx.grid_graph([side_size, side_size], periodic=False)
        g = nx.relabel_nodes(g, lambda x: x[0]*side_size + x[1])
        return g
    elif graph_topology == 'ba':
        return nx.barabasi_albert_graph(n_nodes, m=1, seed=nx_seed)
    elif graph_topology == "tree":
        return nx.random_tree(n_nodes, seed=nx_seed)


def prepare_graph(data_folder: str, topology: str, n_nodes: int):
    """
    Generates the graph adjacency matrix, assigns positions for plotting with Kamada-Kawai layout.
    All data are stored under the folder `{data_folder}/{topology}` path.
    :param data_folder: A string containing the data folder path
    :param topology: he graph topology which can be a string value from `['lattice'. 'ba', 'tree']`, used to create a square lattice, Barabasi-Albert and  random tree graphs respectively.
    :param n_nodes:
    :return: the networkx.Graph object, the position dictionary `{id, [x,y]}`, the adjacency
    torch tensor and the `graph_folder` path string.
    """
    graph_folder = data_folder + str(topology)+'/'
    os.makedirs(graph_folder, exist_ok=True)
    current_graph = default_graph_generator(n_nodes, graph_topology=topology)
    pos = nx.kamada_kawai_layout(current_graph)
    adj_tensor = adjacency_tensor(current_graph).to(torch.uint8)
    torch.save(adj_tensor, graph_folder+'adjacency.pt')
    drivers = torch.tensor(np.array(sorted(maximum_matching_drivers(current_graph))))
    torch.save(drivers, graph_folder+'drivers.pt')

    df = pd.DataFrame.from_dict(pos, orient='index', columns=['x', 'y']).reset_index()
    df.to_csv(graph_folder+'pos.csv', index=False)
    return current_graph, pos, adj_tensor, graph_folder


def prepare_target_states(initial_states: torch.Tensor,
                          adj_matrix: torch.Tensor,
                          graph_folder: str):
    """
    Prepares the random states by evolving the initials states based on continuous cellular
    automaton and the graph adjacency matrix
    :param initial_states: a `torch.Tensor` object containing the initial states in a batch format (
    shape is `[n_samples, n_nodes]`.
    :param adj_matrix: The graph adjacency matrix as a `torch.Tensor ` object.
    :param graph_folder: The string that points to the graph folder.
    :return:
    """
    target_states = torch.cat([cca_state_generator(initial_states[i, :],
                                                   adj_matrix,
                                                   50 + 50 * np.random.randint(0, 10, size=1)[0]
                                                   )
                               for i in range(initial_states.shape[0])
                               ],
                              dim=0
                              )
    torch.save(target_states, str(graph_folder + 'target_states.pt'))


if __name__ == '__main__':
    number_nodes = 1024  # number of nodes per graph
    n_samples = 100  # total samples of initial and target state pairs
    data_path = '../../../data/parameters/ct_lti'  # local folder that the graph data files are stored in

    os.makedirs(data_path, exist_ok=True)

    # Saving and generating the initial states
    init_states = 2 * (torch.rand([n_samples, number_nodes]) - 0.5)
    torch.save(init_states.cpu().detach(), data_path + 'init_states.pt')

    # Square Lattice
    _, _, adjacency_matrix, graph_folder_path = prepare_graph(data_path,
                                                              'lattice',
                                                              number_nodes
                                                              )
    prepare_target_states(init_states, adjacency_matrix, graph_folder_path)

    # Barabasi Albert
    _, _, adjacency_matrix, graph_folder_path = prepare_graph(data_path,
                                                              'ba',
                                                              number_nodes
                                                              )
    prepare_target_states(init_states, adjacency_matrix, graph_folder_path)

    # Random Tree
    _, _, adjacency_matrix, graph_folder_path = prepare_graph(data_path,
                                                              'tree',
                                                              number_nodes
                                                              )
    prepare_target_states(init_states, adjacency_matrix, graph_folder_path)
