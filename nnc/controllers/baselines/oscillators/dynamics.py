import torch
from nnc.controllers.base import ControlledDynamics
from nnc.helpers.torch_utils.numerics import sin_difference, sin_difference_mem


class AdditiveControlKuramotoDynamics(ControlledDynamics):
    def __init__(self,
                 interaction_matrix,
                 coupling_constant,
                 natural_frequencies,
                 frustration_constant = None,
                 driver_mask = None,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        The oscillator dynamics based on:
        'Skardal, P. S., & Arenas, A. (2015). Control of coupled oscillator networks with
        application to microgrid technologies. Science advances, 1(7), e1500339.'
        :param interaction_matrix: this matrix determines the node coupling, often the graph
        adjacency matrix isprovided.
        :param coupling_constant: the coupling constant scalar, preferably positive. Higher
        values synchronize the system without control.
        :param natural_frequencies: The vector of natural frequences, which are intrisic natural
        velocities per node. Vector index coincides with the node index in state vectors.
        :param frustration_constant: A constant that is added inside the sine operation,
        and makes the systems essentially harder to synchronize.
        :param driver_mask: We can use the gain matrix here, but since in the given experiments
        the gain multiplication are handled from he control we can use a binary mask.
        We use this for mainly for tests and debugging.
        :param dtype: the torch data type of the dynamics modules
        :param device: the torch device type of the dynamics modules
        """
        super().__init__(['theta'])
        self.device = device
        self.dtype = dtype
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across batches[
        self.interaction__matrix = interaction_matrix.unsqueeze(0)
        self.n_nodes = interaction_matrix.shape[-1]
        self.driver_mask = driver_mask
        if self.driver_mask is not None and len(self.driver_mask.shape) == 1:
           self.driver_mask = self.driver_mask.unsqueeze(0)
        self.degree_matrix = torch.diag(self.interaction__matrix.sum(-1)[0]).unsqueeze(0)
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative'
        self.frustration_constant = frustration_constant

    def forward(self, t, x, u=None):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. Please ensure the input is not permuted, unless you know
        what you doing.
        :param t: time scalar, which is not used as the model is time invariant.s
        :param u: control vectors. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """
        theta  = x
        dx = self.natural_frequencies

        minuend = theta
        subtrahend = theta

        if self.frustration_constant is not None:
            minuend = minuend + self.frustration_constant

        interaction_term = sin_difference_mem(minuend, subtrahend, self.interaction__matrix)
        coupling_term = self.coupling_constant * interaction_term
        if u is not None:
            control_term = u
            if self.driver_mask is not None:
                control_term = control_term*self.driver_mask
            dx = dx + coupling_term + control_term
        else:
            dx = dx + coupling_term
        return dx
