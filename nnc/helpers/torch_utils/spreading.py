import numpy as np
import torch
import networkx as nx
import math

def participant_selection(n_nodes, proportion=0.5):
    """
    A method that selects participants or driver nodes in epidemic dynamics.
    :param n_nodes: The number of nodes.
    :param proportion: The proportion of participants to nodes.
    :return:
    """
    participants = np.random.choice(list(range(n_nodes)),
                                    size=int(n_nodes*proportion),
                                    replace=False
                                    )
    return participants


def lattice_4square_communities(n_nodes):
    """
    Breaks a square lattice and returns a dictionary of labeled indice lists containing the
    community labels as keys and the node indices per community as values.
    :param n_nodes: The number of nodes in the square lattice
    :return: the dictionary.
    """
    all_ids = list(range(n_nodes))
    side_size = int(n_nodes**(1/2))
    side_half = int(side_size//2)
    manual_communities = dict(
        a=list(filter(lambda i: i % side_size < side_half and int(i // side_size) < side_half, all_ids)),
        b=list(filter(lambda i: i % side_size >= side_half and int(i // side_size) < side_half, all_ids)),
        c=list(filter(lambda i: i % side_size < side_half and int(i // side_size) >= side_half, all_ids)),
        d=list(filter(lambda i: i % side_size >= side_half and int(i // side_size) >= side_half, all_ids)),
    )
    return manual_communities


def liquid_communities(G, n_communities):
    """
    We use liquid communities from `networkx.algorithms.community.asyn_fluidc` to determine
    communities.
    :param G: the `networkx.Graph` object.
    :param n_communities: the number of desired communities.
    :return: a dictionary of labeled indice lists containing the
    community labels as keys and the node indices per community as values.
    """
    return dict(enumerate(nx.algorithms.community.asyn_fluidc(G, n_communities)))


def communities_to_labels(n_nodes, communities):
    """
    For n_nodes nad  a communties dictionary or list of lists returns a list with size 'n_nodes'
    assigning to each node index the corresponding node community.
    :param n_nodes:
    :param communities:
    :return:
    """
    comm_labels = torch.zeros([n_nodes])
    if isinstance(communities, dict):
        comm_index = list(communities.keys())
    else:
        comm_index = range(len(communities))
    i = 1
    for comm in comm_index:
        comm_val = communities[comm]
        for node_id in comm_val:
            comm_labels[node_id] = i
        i += 1
    return comm_labels
