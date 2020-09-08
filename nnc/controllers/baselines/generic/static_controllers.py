import torch
from nnc.controllers.base import BaseController


class NoController(BaseController):
    def __init__(self):
        """
        This controller returns None for control.
        If you plan to use it with custom dynamics, please implement the u=None handler case in
        the implementation of :meth:`nnc.controllers.base.ControlledDynamics.forward`.
        """
        super(NoController, self).__init__()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> None:
        return None


class ZerosController(BaseController):
    def __init__(self):
        """
        This controller always returns a tensor of zeros for control.
        This can be the no control case for additive control in dynamics, i.e. where a linear
        projection of the control is added in the derivative equation, or as a control that
        blocks the effect of a term in multiplicative control terms.
        If you plan to use it with custom dynamics, please implement the u=None handler case in
        the implementation of :meth:`nnc.controllers.base.ControlledDynamics.forward`.
        """
        super(ZerosController, self).__init__()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        sample_shape = [batch_size] + self.control_signal_shape
        return torch.zeros(torch.Size(sample_shape))
