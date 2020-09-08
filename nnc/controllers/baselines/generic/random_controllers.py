import torch
from torch.distributions.distribution import Distribution
from nnc.controllers.base import BaseController


class DistributionSampleController(BaseController):

    def __init__(self, distibution: Distribution, control_signal_shape: list):
        """
        Randomly generated control, following a torch distribution
        :param distibution: The torch distribution object
        :param control_signal_shape: the shape of control without batches
        """
        self.distribution = distibution
        self.control_signal_shape = control_signal_shape

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Samples a batch of control signals randomly from given distributions.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n ]`.
        :return: The control singnal tensor.
        """
        batch_size = x.shape[0]
        sample_shape = [batch_size] + self.control_signal_shape
        return self.distribution.rsample(sample_shape=torch.Size(sample_shape))
