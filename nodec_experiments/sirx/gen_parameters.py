import random

from nnc.helpers.data_generators import random_seed, numpy_seed, nx_seed
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
    `['lattice']`, used to create a square-lattice.
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
    :param topology: he graph topology which can be a string value from `['lattice']`, 
    used to create a squarelattice.
    :param n_nodes: number of nodes
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


def prepare_target_subgraph(n_nodes: int, graph_folder: str):
    """
    Prepares the target subgraph based on the square lattice dimensions.
    :param initial_states: a `torch.Tensor` object containing the initial states in a batch format (
    shape is `[n_samples, n_nodes]`.
    :param adj_matrix: The graph adjacency matrix as a `torch.Tensor ` object.
    :param graph_folder: The string that points to the graph folder.
    :return:
    """
    all_nodes = np.arange(n_nodes)
    target_subgraph_nodes = all_nodes[np.logical_and(all_nodes % 32 < 16, (all_nodes // 32).astype(int) < 16)]
    torch.save(target_subgraph_nodes, str(graph_folder + 'target_subgraph_nodes.pt'))


    
def prepare_initial_infection_nodes(n_nodes, graph_folder):
    """
    Prepares the initial infection on the lattice quadrant that is opposite (across diagonal)
    of the target subgraph based on the square lattice dimensions.
    :param n_nodes: number of nodes in the square lattice.
    :param graph_folder: The folder for saving the initial infection indices into.
    :return: the the initial infection indices.
    """
    all_nodes = np.arange(n_nodes)
    initial_infection_nodes = np.logical_and(all_nodes % 32 > 16, (all_nodes // 32).astype(int) > 16)
    initial_infection_nodes = all_nodes[np.logical_and(initial_infection_nodes, all_nodes % 5 == 0)]
    torch.save(initial_infection_nodes, str(graph_folder + 'initial_infection_nodes.pt'))
    return initial_infection_nodes
    
def prepare_initial_state(initial_infection_nodes, n_nodes, graph_folder):
    inf_0 = torch.zeros([1, n_nodes], device='cpu', dtype=torch.float)
    inf_0[:, initial_infection_nodes] = 1.0
    susc_0 = torch.ones([1, n_nodes], device='cpu', dtype=torch.float) - inf_0
    rec_0 = torch.zeros([1, n_nodes], device='cpu', dtype=torch.float)
    cont_0 = torch.zeros([1, n_nodes], device='cpu', dtype=torch.float)
    x0 = torch.cat([inf_0, susc_0, rec_0, cont_0], dim=-1).detach()
    torch.save(x0, str(graph_folder + 'initial_state.pt'))
    
    
if __name__ == '__main__':
    number_nodes = 1024  # number of nodes per graph
    n_samples = 100  # total samples of initial and target state pairs
    data_path = '../data/parameters/sirx/'  # local folder that the graph data files are stored in

    os.makedirs(data_path, exist_ok=True)


    # Square Lattice
    _, _, adjacency_matrix, graph_folder_path = prepare_graph(data_path,
                                                              'lattice',
                                                              number_nodes
                                                              )
    prepare_target_subgraph(number_nodes, graph_folder_path)
    init_infection_nodes = prepare_initial_infection_nodes(number_nodes, graph_folder_path)
    prepare_initial_state(init_infection_nodes, number_nodes, graph_folder_path)
    
    dynamics_parameters = dict(budget = 600,
                               infection_rate = 6,
                               recovery_rate=1.8
                              ) 
    torch.save(dynamics_parameters, graph_folder_path + 'dynamics_parameters.pt')