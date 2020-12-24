import numpy as np
import torch
import random

random_seed = 1
numpy_seed = 1
nx_seed = 1
torch_seed = 1
torch_device = 'cpu'
torch_dtype = torch.float


def uniform_state_generator(samples: int, n_nodes: int):
    """
    Generates a uniformly distributed sample in `[0,1]`
    of size `samples` number of states for `n_nodes`.
    :param samples: The number of required samples.
    :param n_nodes: The number of nodes
    :return: A matrix with shape `[samples, n_nodes]`
    """
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    return torch.rand([samples, n_nodes], dtype=torch_dtype, device=torch_device)


def normal_state_generator(samples: int, n_nodes: int):
    """
    Generates a normally distributed sample with `(mean=0,std=1)`
    of size `samples` number of states for `n_nodes`.
    :param samples: The number of required samples.
    :param n_nodes: The number of nodes
    :return: A matrix with shape `[samples, n_nodes]`
    """
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    return torch.randn([samples, n_nodes], dtype=torch_dtype, device=torch_device)


def cca_state_generator(initial_state: torch.Tensor,
                        adjacency_matrix: torch.Tensor,
                        iterations: int):
    """
    A continuous cellular automaton used to generated patterns of target states on graph neighborhs.
    Used in CTI-LTI experiments.
    :param initial_state: The initial state vector to evolve.
    :param adjacency_matrix: The graph adjacency matrix of shape `[n,n]`
    :param iterations: Number of iterations for evolution. The more iterations the more similar
    the target states will be across all graph nodes.
    :return: The target state tensor, that has same dimensionality as `initial_state`.
    """
    x = initial_state.unsqueeze(0)
    for i in range(iterations):
        y = adjacency_matrix * x
        y_max = y.max(-1).values
        y_min = y.min(-1).values
        to_add = y_max
        bindex = y_max.abs() < y_min.abs()
        to_add[bindex] = y_min[bindex]
        x = x * (1 - 0.05) + to_add * 0.05
    return x
