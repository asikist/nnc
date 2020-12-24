import torch
from nnc.controllers.base import BaseController


class KuramotoFeedbackControl(BaseController):
    """
    A simple feedback controller for 2-pi periodic dynamics according to:
    'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks
    with application to microgrid technologies. Science advances, 1(7), e1500339.'
    """

    def __init__(self, gain_vector, target_state=0):
        """
        A simple feedback controller based on the description of the paper:
        'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks with
        application to microgrid technologies. Science advances, 1(7), e1500339.'
        :param gain_vector: The diagonal of the driver/gain matrix. We use element-wise
        multiplication instead of inner product to save some memory, as the method assumes non-zero
        diagonals.
        :param target_state: The target state can be any desired state.
        """
        super().__init__()
        self.target_state = target_state
        self.gain_vector = gain_vector

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Control method
        :param t: the time, since the controller is state-feedback, time is not used.
        :param x: the state, which is used to caculate the control signal.
        :return:
        """
        u = self.gain_vector*torch.sin(self.target_state-x)
        return u
