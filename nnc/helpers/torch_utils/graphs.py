import random
from typing import Collection

import networkx as nx
import numpy as np
import torch

from nnc.helpers.data_generators import random_seed, numpy_seed, torch_seed, torch_dtype, \
    torch_device
from nnc.helpers.error_handling import shape_expectation_failed
from nnc.helpers.graph_helper import nx_maximum_matching
from nnc.helpers.torch_utils.indexing import multi_unsqueeze

"""
Some graph utilities which were developed to check the experiments.
"""


def calc_neighborhood_state_tensor(state_tensor, interaction_mask, interaction_mask_index):
    """
    Calculates the neighborhood state tensor.
    This tensor is of shape: `[:, m, n_nodes, d]` where `m` are the state variables, `n_nodes` the number of
    nodes and `d` is the maximum degree
    :param state_tensor: The state tensor, often of shape `[:, m, n_nodes]`
    :param interaction_mask: the interaction mask, of shape `[n_nodes, d]`
    :param interaction_mask_index: the interaction mask index, of shape `[n_nodes, d]`
    :return:
    """
    assert len(interaction_mask.shape) == len(interaction_mask_index.shape) == 2, \
        shape_expectation_failed("Interaction mask and interaction mask index are not matrices.",
                                 '',
                                 dict(interaction_mask=interaction_mask,
                                      interaction_mask_index=interaction_mask_index)
                                 )
    assert interaction_mask.shape == interaction_mask_index.shape, \
        shape_expectation_failed("Interaction mask and interaction mask index should have "
                                 "the same shape",
                                 '',
                                 dict(interaction_mask=interaction_mask,
                                      interaction_mask_index=interaction_mask_index)
                                 )
    if len(state_tensor.shape) < 3:
        return state_tensor[:, interaction_mask_index.unsqueeze(0)] * interaction_mask.unsqueeze(0)
    else:
        shape_diff = len(state_tensor.shape) - 2
        total_slices = [slice(None)] * (shape_diff + 1)
        total_slices.append(interaction_mask_index)
        return state_tensor[total_slices] * multi_unsqueeze(interaction_mask, shape_diff)
    # TODO furher develop to optimize SIRX methods
    # sel_states = states[:, inds]
    # coupling = sel_states * mask
    # return coupling


def calc_symmetric_interaction_mask(interaction_matrix, descending=True):
    """
    This method generates the  "interaction mask" tensor and indices based on an
    interaction matrix.
    The interaction mask can be used to generate the neighbor state tensor.


    :param interaction_matrix:  Essentially an interaction matrix is a square matrix that determines
       interactions between nodes in the form of nodes. For `n_nodes` nodes, the interaction matrix has shape
       `[n_nodes, n_nodes]`.
    :param descending, whereas the mask contains zero elements on the left or right side of the
    column dimension.
    :return: Two tensors:
    - The first is the interaction mask, i.e. the sorted interaction matrix
    across the last dimension.
    - The second are the interaction indices, i.e. the indices of the

    An interaction mask can be applied on a state vector of shape `[m, n_nodes]`, of `m` state
    variables and `n_nodes` nodes.
    The result of applying such mask is an `[m, n_nodes, d]` tensor, which is referred to as the
    neighbor state tensor.
    Please look at :meth:`nnc.helpers.torch_utils.learning.architectures.gnns
    .neighborhood_state_tensor` for more details.
    For the symmetric case, i.e. undirected and unweighted graph, the max degree is calculated
    as the maximum count of non-zero elements across a dimension.


    where `d` is the max network degree.
    Essentially, this result is a tensor which contains the state variable values of first degree
    neighbors

    """
    max_degree = (interaction_matrix != 0).sum(-1).max().to(torch.long).item()  # assume symmetric
    # matrix
    # absolute used for compatibility with negative relationships.
    mask, inds = interaction_matrix.abs().sort(-1, descending=descending)
    mask = torch.gather(interaction_matrix, -1, inds)
    index_slice = [slice(-max_degree, None), slice(None, max_degree)][descending]
    return mask[:, index_slice], inds[:, index_slice]


class GraphDirections:
    """
    Direcition of a graph. Used to decide how to calculate degree on adjacency matrix.
    Bidirectional graphs have both in and out degree.
    """
    undirected = 0
    row_col = 1
    col_row = 2
    bidirectional = 3


def get_max_in_degree(adjacency_matrix, direction: int = GraphDirections.undirected):
    """
    Get max indegree based on an adjacency matrix.
    :param adjacency_matrix:  the adjacency matrix
    :param direction:
    :return:
    """
    if direction in {GraphDirections.undirected, GraphDirections.row_col}:
        # sum across rows and preserve columns
        return (adjacency_matrix != 0).sum(0).max().item()
    elif direction == GraphDirections.col_row:
        return (adjacency_matrix != 0).sum(1).max().item()
    elif direction == GraphDirections.bidirectional:
        return max([(adjacency_matrix != 0).sum(0).max().item(), (adjacency_matrix != 0).sum(
            0).max().item()])
    else:
        raise ValueError('Unknown  graph direction specification provided! '
                         'Please check the graph `GraphDirections` class for setting the direction '
                         'variable properly!')


def maximum_matching_drivers(G: nx.Graph):
    """  Alias for `nnc.helpers.graph_helper.nx_maximum_matching`.
    The maximum method used to determine drivers nodes for CT-LTI systems according to:
    Liu, Y. Y., Slotine, J. J., & Barabási, A. L. (2011). Controllability of complex networks.
    nature, 473(7346), 167-173.
    :param G: The `networkx.Graph` objext
    :return: The sorted list of node indices that can be used as drivers.
    """
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    return nx_maximum_matching(G)


def drivers_to_tensor(n_nodes, drivers: Collection):
    """
    Converts a list of driver node with indices over n_nodes to a binary driver matrix of shape [
    n_nodes, n_drivers]
    :param n_nodes: The number of nodes in the graph
    :param drivers: the list of drivers with node index
    :return:
    """
    driver_matrix = torch.zeros([n_nodes, len(drivers)], dtype=torch_dtype, device=torch_device)
    for i, j in enumerate(sorted(drivers)):
        driver_matrix[j, i] = 1.0
    return driver_matrix


def adjacency_tensor(G: nx.Graph):
    """
    Gets a `torch.Tensor` dense adjacency matrix from a `networkx.Graph` object.
    :param G:
    :return:
    """
    return torch.tensor(np.array(nx.adjacency_matrix(G).todense()), device=torch_device,
                        dtype=torch_dtype)
