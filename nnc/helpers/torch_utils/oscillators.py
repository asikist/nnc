import numpy as np
import torch
from nnc.helpers.torch_utils.indexing import get_off_diagonal_elements
import math

"""
Basic oscillator utilities.
"""


def order_parameter_cos(x):
    """
    The order parameter calculated based on the cosine formula in the paper.
    :param x: the state of the oscillators
    :return: The r value
    """
    n = x.shape[-1]
    diff = torch.cos(x.unsqueeze(-1) - x.unsqueeze(-2))
    sum_diff = (diff).sum(-1).sum(-1)
    r = (1 / n) * (sum_diff ** (1 / 2))
    return r


def calculate_steady_state(natural_frequencies,
                           coupling_strength,
                           adjacency_matrix
                           ):
    """
    Steady state calculation according to:
    'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks with
    application to microgrid technologies. Science advances, 1(7), e1500339.'
    :param natural_frequencies:
    :param coupling_strength:
    :param adjacency_matrix:
    :return: the steadu state
    """
    natural_frequencies = natural_frequencies.unsqueeze(-1)
    laplacian_matrix = adjacency_matrix.sum(-1).diag() - adjacency_matrix
    ldagger = torch.pinverse(laplacian_matrix)
    steady_state = (1 / coupling_strength) * \
                   (ldagger @ natural_frequencies)
    return steady_state.squeeze(-1)


def calculate_steady_state_jaccobian(natural_frequencies,
                                     coupling_strength,
                                     adjacency_matrix,
                                     steady_state=None
                                     ):
    """
    Jaccobian calculation based on steady state according to:
    'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks with
    application to microgrid technologies. Science advances, 1(7), e1500339.'
    :param natural_frequencies:
    :param coupling_strength:
    :param adjacency_matrix:
    :return: the steady state jaccobian
    """
    if steady_state is None:
        steady_state = calculate_steady_state(
            natural_frequencies,
            coupling_strength,
            adjacency_matrix
        )
    df = coupling_strength * torch.cos(
        steady_state.unsqueeze(-2) -
        steady_state.unsqueeze(-1)
    ) * adjacency_matrix

    diag_inds = list(range(0, adjacency_matrix.shape[-1]))
    df[diag_inds, diag_inds] = -df.sum(-1)
    return df


def calc_driver_vector(steady_state_jaccobian, adjacency_matrix, estimation_margin=0.1):
    """
    Gain vector calculation according to:
    'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks with
    application to microgrid technologies. Science advances, 1(7), e1500339.'
    :param steady_state_jaccobian:
    :param adjacency_matrix:
    :param estimation_margin: the buffer margin based on the paper.
    :return:
    """
    off_diagonal_elements = get_off_diagonal_elements(steady_state_jaccobian) - \
                            estimation_margin * adjacency_matrix
    return (off_diagonal_elements.abs() - off_diagonal_elements).sum(-1).float()


def calc_driver_nodes(steady_state_jaccobian, adjacency_matrix, estimation_margin=0.1):
    off_diagonal_elements = get_off_diagonal_elements(steady_state_jaccobian) - \
                            estimation_margin * adjacency_matrix
    neg_off_diag = (off_diagonal_elements < 0) & (adjacency_matrix > 0)
    driver_nodes = set(torch.nonzero(neg_off_diag)[:, 0].cpu().detach().numpy().tolist())
    return driver_nodes


def calculate_frustration(fun, derivative_fun):
    """
    Frustration value based on the given function and its derivative.
    :param fun: the function to calculate the frustration
    :param derivative_fun: The derivative of the provided function.
    :return:
    """
    return fun(0) / math.sqrt(2) / derivative_fun(0)


def calculate_kuramoto_frustration(frustration_constant=0):
    """
    Frustration calculation for kuramoto model based on frustration constant value.
    :param frustration_constant: the frustration constant.
    :return:
    """
    fun = lambda x: math.sin(frustration_constant)
    derivative_fun = lambda x: math.cos(frustration_constant)
    return calculate_frustration(fun, derivative_fun)
