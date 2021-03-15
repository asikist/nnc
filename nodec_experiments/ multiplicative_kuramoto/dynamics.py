import torch
from nnc.controllers.base import ControlledDynamics


class ForwardKuramotoDynamics(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,
                 driver_matrix,
                 coupling_constant,
                 natural_frequencies,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Kuramoto primal system
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usuallu "cpu" or "cuda:0"
        """
        super().__init__(['theta'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across
        # batch dimension
        self.adjacency_matrix = adjacency_matrix.unsqueeze(0)
        self.n_nodes = adjacency_matrix.shape[-1]
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.degree_matrix = torch.diag(self.adjacency_matrix.sum(-1)[0]).unsqueeze(0)
        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative-forward'

    def forward(self, t, x, u):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. In this case the state is the angle of the
        system and the angular velocity.
        :param t: time scalar, which is not used as the model is time invariant.
        :param u: control vector. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """

        if len(x.shape) > 2:
            theta = x[:, 0, :]
        else:
            theta = x

        dx = self.natural_frequencies

        # calculation of the interaction term: F
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        interaction_term = (cos_theta * torch.matmul(self.adjacency_matrix, sin_theta)) \
            .squeeze(-1)
        interaction_term = interaction_term - (sin_theta * torch.matmul(self.adjacency_matrix,
                                                                        cos_theta)).squeeze(-1)

        coupling_term = (self.coupling_constant / self.n_nodes) * interaction_term
        if u is not None:
            control_term = torch.matmul(self.driver_matrix, u.unsqueeze(-1)).squeeze(-1)
            dx = dx + control_term * coupling_term
        else:
            dx = dx + coupling_term

        return dx


class BackwardKuramotoDynamics(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,  # oscillator coupling
                 driver_matrix,  # identity
                 coupling_constant,  # K
                 natural_frequencies,  # omega
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Kuramoto primal system
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usually "cpu" or "cuda:0"
        """
        super().__init__(['theta', 'p'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across
        self.interaction_matrix = adjacency_matrix.unsqueeze(0)
        self.n_nodes = adjacency_matrix.shape[-1]
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.degree_matrix = torch.diag(self.interaction_matrix.sum(-1)[0]).unsqueeze(0)
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative-backward'
        self.k_by_N = self.coupling_constant / self.n_nodes

    def forward(self, t, x, u):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. Please ensure the input is not permuted, unless you know
        what you doing.
        :param t: time scalar, which is not used as the model is time invariant.s
        :param u: control vectors. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """
        p = x[:, 0, :]
        theta = x[:, 1, :]
        dp = self._adjoint_backward(p, theta, t, u)
        dx = dp
        return dx

    def _adjoint_backward(self, p, theta, t, u):
        """
        Evaluation of the adjoint system dp/dt
        :param p: current adjoint vector.
        :param theta: current theta vector.
        :param t: time scalar (note that the current dynamics has no explicit
                  time dependence)
        :param u: control vector
        :return: dp/dt, du/dt
        """
        k_by_n_u = self.k_by_N * u

        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        interaction_term_1_part_a = (cos_theta * torch.matmul(self.interaction_matrix,
                                                              cos_theta)).squeeze(-1)
        interaction_term_1_part_b = (sin_theta * torch.matmul(self.interaction_matrix,
                                                              sin_theta)).squeeze(-1)
        interaction_term_1 = interaction_term_1_part_a + interaction_term_1_part_b

        control_vector_1 = k_by_n_u * p
        sum_term_1 = control_vector_1 * interaction_term_1

        # sum, term 2
        control_vector_2 = self.k_by_N * u  # optimize here
        interaction_term_2_part_a = (cos_theta * torch.matmul(self.interaction_matrix,
                                                              p.unsqueeze(-1) * cos_theta)
                                     ).squeeze(-1)
        interaction_term_2_part_b = (sin_theta * torch.matmul(self.interaction_matrix,
                                                              p.unsqueeze(-1) * sin_theta)
                                     ).squeeze(-1)
        interaction_term_2 = interaction_term_2_part_a + interaction_term_2_part_b

        sum_term_2 = control_vector_2 * interaction_term_2

        dp = sum_term_1 - sum_term_2

        return dp
