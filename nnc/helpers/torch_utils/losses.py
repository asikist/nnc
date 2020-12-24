import torch
import numpy as np

"""
Losses to be used with evaluators in `nnc.helpers.torch_utils.evaluators`
"""


class FinalStepMSE(torch.nn.Module):

    def __init__(self, x_target, total_time):
        """
        MSE loss when t = T, based on target state.
        :param x_target: The target state.
        :param total_time: The total_time.
        """
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.x_target = x_target
        self.total_time = total_time

    def forward(self, t, x):
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach().numpy()
        if np.allclose(t - self.total_time, 0):
            return self.criterion(x, self.x_target)
        else:
            return None
