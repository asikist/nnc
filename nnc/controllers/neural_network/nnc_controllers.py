import torch
from nnc.controllers.base import ControlledDynamics, BaseController


class NeuralNetworkController(BaseController):

    def __init__(self, neural_net: torch.nn.Module):
        """
        Neural network wrapper for NNC.
        Provide the neural network as a submodule.
        """
        super().__init__()
        self.neural_net = neural_net

    def forward(self, t, x) -> torch.Tensor:
        """
        Wrapper method for the neural network.
        It is important that time and state tensors are provided to the neural network,
        and have the required dimensionality and values for control.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: A tensor containing control values, shape: `[b, ?, ?]`
        """
        return self.neural_net(t, x)


class NNCDynamics(torch.nn.Module):
    def __init__(self,
                 underlying_dynamics: ControlledDynamics,
                 neural_network: torch.nn.Module,
                 ):
        """
        A constuctor that couples the controlled dynamics with the neural network.
        :param underlying_dynamics: A class implementing :class:`nnc.controllers.base.ControlledDynamics`
        :param neural_network: A neural network implementing a torch module, with inputs and
        outputs described in  :method:`nnc.controllers.base.NeuralNetworkController`
        """
        super().__init__()
        # assign nnc to the wrapper, may be considered redundant but for the sake of clarity
        self.nnc = NeuralNetworkController(neural_network)
        self.underlying_dynamics = underlying_dynamics
        # for ease of use, so that one can access the same pointer faster
        self.state_var_list = underlying_dynamics.state_var_list

    def forward(self, t, x):
        """
        Calculates the derivative or **amount of change** under neural network control for the
        given dynamics.
        Preserves gradient flows for training.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
        shape: `[b, m, n_nodes]`
        """

        u = self.nnc(t, x)
        dx = self.underlying_dynamics(t=t, u=u, x=x)
        return dx
