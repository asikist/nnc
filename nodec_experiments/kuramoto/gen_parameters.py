import numpy as np
import pandas as pd
import torch
import math

import random
import os

import networkx as nx

from nnc.helpers.torch_utils.graphs import  adjacency_tensor
from nnc.helpers.torch_utils.oscillators import calculate_steady_state, \
    calculate_steady_state_jaccobian, calc_driver_nodes, calc_driver_vector

random_seed = 1
numpy_seed = 1
nx_seed = 1
gt_seed = 1
pt_seed = 1

def default_graph_generator(n_nodes, graph_topology):
    """
    Graph generator for the experiments
    :param n_nodes: the requested number of nodes
    :param graph_topology: The graph topology, an erdos renyi graph generated as described
    in the work of Skardai and Arenas 2015:
    https://advances.sciencemag.org/content/advances/1/7/e1500339.full.pdf

    The number of nodes is set to 1024.
    :return:
    """
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    if  graph_topology == 'erdos_renyi':
        mean_degree = 6
        G = nx.erdos_renyi_graph(n_nodes, p=(mean_degree / (n_nodes - 1)), seed=nx_seed)
    return G


def prepare_graph(data_folder, topology, n_nodes):
    """
    Preparation of graph data required for the experiments
    :param data_folder: The path to the data folder for all graphs
    :param topology: The topology of the requested graph to generate
    :param n_nodes: The number of nodes
    :return:
    """
    graph_folder = data_folder + str(topology)+'/'
    os.makedirs(graph_folder, exist_ok=True)
    G = default_graph_generator(n_nodes, graph_topology=topology)
    pos = nx.kamada_kawai_layout(G)
    adj_tensor = adjacency_tensor(G).to(torch.uint8)
    torch.save(adj_tensor, graph_folder + 'adjacency.pt')

    df = pd.DataFrame.from_dict(nx.kamada_kawai_layout(G), orient='index',
                                columns=['x', 'y']).reset_index()
    df.to_csv(graph_folder + 'pos.csv', index=False)
    return G, pos, adj_tensor, graph_folder


def prepare_dynamics_parameters(data_folder, n_nodes):
    """
    Generate generic dynamic parameters for the oscillator dynamics
    :param data_folder: the path where the data are stored
    :param n_nodes: number of nodes in graph
    :return:
    """
    natural_frequencies = torch.rand(n_nodes)*2*math.sqrt(3) - math.sqrt(3)
    coupling_constants = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    frustration_constants = torch.tensor([0.0, 0.6, 0.9])
    torch.save(natural_frequencies, data_folder + 'nominal_angular_velocities.pt')
    torch.save(coupling_constants, data_folder + 'coupling_constants.pt')
    torch.save(frustration_constants, data_folder + 'frustration_constants.pt')
    return natural_frequencies, coupling_constants, frustration_constants

def prepare_graph_dynamics_parameters(adjacency_matrix,
                                      natural_frequencies,
                                      estimation_margin,
                                      graph_folder
                                      ):
    """
    Prepares and stores parameters and metadata related to the generation of the graph experiments.
    :param adjacency_matrix: The adacency matrix of the graph
    :param natural_frequencies: the list of natural frequencies
    :param estimation_margin: the estimation margin for driver nodes (or error buffer) as
    referred to in:
    https://advances.sciencemag.org/content/advances/1/7/e1500339.full.pdf
    :param graph_folder: The graph folder to save the data
    :return:
    """
    graph_dynamics_folder = graph_folder + os.path.sep + 'dynamics_parameters' + os.path.sep
    for coupling_constant in coupling_constants.detach().cpu().numpy():
        coupling_constant_folder = graph_dynamics_folder + 'coupling_' + str(coupling_constant) +\
                                   os.path.sep
        os.makedirs(coupling_constant_folder, exist_ok=True)

        steady_state = calculate_steady_state(natural_frequencies,
                                              coupling_constant,
                                              adjacency_matrix
                                             )
        torch.save(steady_state, coupling_constant_folder+'unnormalized_steady_state.pt')
        steady_state = steady_state % (2 * math.pi)
        torch.save(steady_state, coupling_constant_folder+'steady_state.pt')

        steady_state_jaccobian = calculate_steady_state_jaccobian(
            natural_frequencies,
            coupling_constant,
            adjacency_matrix,
            steady_state=steady_state
        )
        torch.save(steady_state_jaccobian, coupling_constant_folder+'steady_state_jaccobian.pt')

        driver_nodes = calc_driver_nodes(steady_state_jaccobian,
                                         adjacency_matrix,
                                         estimation_margin=estimation_margin
                                         )
        pd.Series(list(driver_nodes), name='diver_index').to_csv(
                                           coupling_constant_folder+'driver_nodes.csv',
                                             index=False, header=True)

        driver_vector = calc_driver_vector(steady_state_jaccobian,
                                           adjacency_matrix,
                                           estimation_margin=estimation_margin
                                           )
        torch.save(driver_vector, coupling_constant_folder+'driver_vector.pt')


if __name__ == '__main__':
    torch.manual_seed(pt_seed)
    n_nodes = 1024
    data_path = '../../../data/parameters/kuramoto/'
    os.makedirs(data_path, exist_ok=True)
    estimation_margin = 0.1

    ### Generic Dynamics Parameters ###
    natural_frequencies, coupling_constants, frustration_constants = prepare_dynamics_parameters(
        data_path, n_nodes)


    ### Random Graph ###
    G, pos, adjacency_matrix_rg, graph_folder = prepare_graph(data_path, 'erdos_renyi', n_nodes)
    prepare_graph_dynamics_parameters(adjacency_matrix_rg.float(),
                                      natural_frequencies,
                                      estimation_margin,
                                      graph_folder
                                      )





