from numbers import Number
from typing import Union
from typing import Iterable

import torch
from nnc.controllers.base import ControlledDynamics


class ContinuousTimeInvariantDynamics(ControlledDynamics):
    def __init__(self,
                 interaction__matrix,
                 driver_matrix,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Coontinuous time time-invariant linear dynamics of the form: `dx/dt = Ax + Bu`.
        :param interaction__matrix: The interaction matrix `A`, that determines how states of `n_nodes`
        nodes interact, shape `n_nodes x n_nodes`.
        :param driver_matrix: The driver matrix B, which determines how `k` control signals are
        applied in the linear dynamics, shape `n_nodes x k`.
        :param dtype: torch datatype of the calculations
        :param device: torch device of the calculations, usually "cpu" or "cuda:0"
        """
        super().__init__(['x'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # un-squeeze matrices in the first dimension so that operation broadcasts across batches[
        self.interaction__matrix = interaction__matrix.unsqueeze(0)
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.label = 'linear'

    def forward(self, t: Union[(torch.Tensor, Number)],
                x: Union[torch.Tensor, Iterable[torch.Tensor]],
                u: torch.Tensor = None
                ):
        """
        Evaluation of the derivative or **amount of change** for controlled continuous-time
        time-invariant linear dynamics.
        :param x: current state values for nodes. Please ensure the input is not permuted, unless you know
        what you doing.
        :param t: time scalar, which is not used as the model is time invariant.
        :param u: control vectors. In this case please confirm  it has proper dimensionality such
        that
        torch.matmul(driver_matrix, u) is possible.
        :return: the derivative tensor.
        """
        if not isinstance(x, torch.Tensor) and isinstance(x, Iterable):
            # in case somehow state variable batches passed as a tuple or list.
            x = torch.stack(list(x))

        # batch matrix multiplication, broadcasting at dimension 0 (batch dimension)
        # a for loop and normal matrix multiplication can be used instead.

        # calculate  matrix product `<A,x>`
        dx = torch.matmul(self.interaction__matrix, x.unsqueeze(-1)).squeeze(-1)

        if u is not None:
            # if control signals are provided, calculate matrix product `<B,u>`
            control_term = torch.matmul(self.driver_matrix, u.unsqueeze(-1)).squeeze(-1)
            # add both parts
            dx += control_term

        return dx
