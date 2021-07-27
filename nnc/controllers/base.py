from abc import abstractmethod, ABC

import torch
from torch import nn


class BaseController(ABC, nn.Module):
    def __init__(self):
        """
        The template for a controller class.
        """
        super().__init__()

    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        A method implementing the control calculation based on either or both time and state
        values.
        The following notation is used for shapes:
        `b` dimensions for batch size, `m` dimensions for state variables and `n_nodes` dimensions for
        number of nodes.
        Explicit dimension assignment of inputs and outputs for specific functionality are
        defined in the corresponding implementation.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: A tensor containing control values, shape: `[b, ?, ?]`

        A controller  that takes as input a `b x 1` dimensional time `t` tensor and an `b x m x
        n_nodes`-dimensional state `x` to calculate control signals `u` of arbitrary dimensionality.
        """
        pass


class ControlledDynamics(torch.nn.Module):
    def __init__(self, state_var_list: list):
        """
        The template for control dynamics, which showcases how the controlled dynamics code is
        structured to interface seamlessly with this package.
        :param state_var_list: A list of the state variable labels/names for utility purposes.

        E.g. in SIS dynamics model our state is a matrix of dimensions:
        `m x n_nodes`, for `n_nodes` nodes and `m` = 2 state variables:
        `state_var_list = [susceptible, infected]`.
        This indicates that in our state vector, `x[:,0,:]` contains the **susceptible** values
        across all batch samples for all nodes, whereas `x[:,1:]` contains the **infected**
        values across all batch samples for all nodes.
        """
        super().__init__()
        self.state_var_list = state_var_list

    @abstractmethod
    def forward(self, t, x, u=None) -> torch.Tensor:
        """
        The abstract controlled dynamics forward method, used to calculate the derivative under
        control.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`.
        :param u: A tensor containing the control values, can be of arbitrary shape.
        E.g. for a scalar control signal per node, shape: `[b, m, n_nodes]`.
        :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
        shape: `[b, m, n_nodes]`
        """
        # TODO: memory efficient implementation for heterogenous nodes:
        # i.e. state variables that apply to or are shared by a subset of nodes.
        pass
