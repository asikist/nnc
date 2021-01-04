import torch
from torchdiffeq import odeint_adjoint

"""
Utilities for numerical operations in torch.
"""


def simpson(f, a, n, h, progress_bar=None):
    """
    Simpson rule for integral approximation of function `f` for `n_nodes` steps/strips of width `h`,
    starting from point `a`.
    :param f: the function to evaluate the integral for
    :param a: left bound for area evaluation
    :param n: number of strips
    :param h: strip size
    :param progress_bar: a method that wraps the `range` generator and produces a progress bar.
    e.g. use the package `tqdm`, and the progress bar will be called as: `tqdm(range(n_nodes))`
    :return: the integral value
    """
    res = 0
    progress_range = range(n)
    if progress_bar is not None:
        progress_range = progress_bar(progress_range)
    for i in progress_range:
        xi = a + i * h
        xip1 = a + (i + 1) * h
        res += 1. / 6 * (xip1 - xi) * (f(xi) + 4 * f(0.5 * (xi + xip1)) + f(xip1))
    return res


def rms_norm(tensor):
    """
    Root mean squared norm over a tensor.
    :param tensor:
    :return:
    """
    return tensor.pow(2).mean().sqrt()


def make_norm(state):
    """
    Augmented state norm
    :param state: the state to augment
    :return:
    """
    state_size = state.numel()

    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))

    return norm


def faster_adj_odeint(func, y0, t, **kwargs):
    """
    faster `torch.diffeq.adjoint` based on:
    `https://github.com/patrick-kidger/FasterNeuralDiffEq/`
    :param func: the dynamics function
    :param y0: the intitial state
    :param t: the time interval, often a linspace
    :param kwargs: keyword arguments for `torchdiffeq.odeint_adjoint`
    :return:
    """
    #
    if kwargs is None:
        kwargs = dict()
    if 'adjoint_options' not in kwargs:
        kwargs['adjoint_options'] = dict()
    kwargs['adjoint_options']['norm'] = make_norm(y0)
    return odeint_adjoint(func=func, y0=y0, t=t,
                          **kwargs)


def sin_difference(a, b, adjacency_matrix):
    """
    The difference of coupled states in an oscillator model.
    Vectorized.
    :param a: the vector a
    :param b: the vector b
    :param adjacency_matrix: The adjacency matrix
    :return: The vector y with elements calculated as: \sum_j sin(a_j - b_i).
    """
    # calculates sum_j sin(a_j-b_i), but better check tests to clarify. It works as equations are
    # written in paper.
    if len(a.shape) == 1 or a.shape[-1] != 1:
        a = a.unsqueeze(-1)

    if len(b.shape) == 1 or b.shape[-1] != 1:
        b = b.unsqueeze(-1)

    if len(a.shape) > 2 and len(adjacency_matrix.shape) == 2:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)

    sin_a = torch.sin(a)
    cos_a = torch.cos(a)

    if torch.all(a == b):
        sin_b = sin_a
        cos_b = cos_a
    else:
        sin_b = torch.sin(b)
        cos_b = torch.cos(b)

    res = cos_b * (adjacency_matrix @ sin_a) - (adjacency_matrix @ cos_a) * sin_b
    return res.squeeze(-1)


def sin_difference_mem(a, b, adjacency_matrix):
    """
    The difference of coupled states in an oscillator model.
    Vectorized, faster but costs more memory.
    :param a: the vector a
    :param b: the vector b
    :param adjacency_matrix: The adjacency matrix
    :return: The vector y with elements calculated as: \sum_j sin(a_j - b_i).
    """
    # calculates sum_j sin(a_j-b_i), but better check tests to clarify. It works as equations are
    # written in paper.
    if len(a.shape) > 1:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
    res = (torch.sin(a.unsqueeze(-2) - b.unsqueeze(-1)) * adjacency_matrix).sum(-1)
    return res.squeeze(-1)


def cos_difference(a, b, adjacency_matrix):
    """
    The difference of coupled states in an oscillator model.
    Vectorized.
    :param a: the vector a
    :param b: the vector b
    :param adjacency_matrix: The adjacency matrix
    :return: The vector y with elements calculated as: \sum_j cos(a_j - b_i).
    """
    if len(a.shape) == 1 or a.shape[-1] != 1:
        a = a.unsqueeze(-1)

    if len(b.shape) == 1 or b.shape[-1] != 1:
        b = b.unsqueeze(-1)

    if len(a.shape) > 2 and len(adjacency_matrix.shape) == 2:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)

    sin_a = torch.sin(a)
    cos_a = torch.cos(a)

    if torch.all(a == b):
        sin_b = sin_a
        cos_b = cos_a
    else:
        sin_b = torch.sin(b)
        cos_b = torch.cos(b)

    res = cos_b * (adjacency_matrix @ cos_a) + (adjacency_matrix @ sin_a) * sin_b
    return res.squeeze(-1)


def cos_difference_mem(a, b, adjacency_matrix):
    """
    The difference of coupled states in an oscillator model.
    Vectorized, faster but costs more memory.
    :param a: the vector a
    :param b: the vector b
    :param adjacency_matrix: The adjacency matrix
    :return: The vector y with elements calculated as: \sum_j cos(a_j - b_i).
    """
    if len(a.shape) > 1:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)

    res = (torch.cos((a.unsqueeze(-2) - b.unsqueeze(-1))) * adjacency_matrix).sum(-1)
    return res.squeeze(-1)
