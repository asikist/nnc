import torch

from nnc.helpers.torch_utils.numerics import simpson
from nnc.helpers.torch_utils.expm.expm import expm as torch_expm
from nnc.controllers.base import BaseController


class ControllabiltyGrammianController(BaseController):

    def __init__(self, alpha, beta, t_infs, x0s, x_infs, simpson_evals,
                 progress_bar=None, use_inverse=False):
        """
        Optimal Control baseline using the controllability grammian, as showcased in the work of:
        **Yan, G., Ren, J., Lai, Y. C., Lai, C. H., & Li, B. (2012). Controlling complex
        networks: How much energy is needed?. Physical review letters, 108(21), 218703.**
        :param alpha: the influence/adjacency matrix that contains nodal interactions.
        :param beta: the driver matrix used to determine driver nodes
        :param t_infs: end times for batched version over many end times. Initial times are
        always considered to be 0.
        :param x0s: initial states
        :param x_infs: target states
        :param simpson_evals: number of simplson intervals/strips used for intgral of grammin term
        :param progress_bar: The function to wrap the `range` generator and produce a progress
        bar over simpson evaluators.
        :param use_inverse: whether to use :meth:`torch.inverse` or `torch.solve` for the inverse
        term calculations involved in the controllability grammian estimation.
        """
        super().__init__()
        self.A = alpha  # n_nodes x n_nodes
        self.B = beta  # n_nodes x m

        self.Tf = t_infs

        self.expA = torch_expm(self.A * self.Tf).unsqueeze(0)
        self.N = self.A.shape[0]
        self.f = lambda t: torch.chain_matmul(torch_expm(self.A * t), self.B,
                                              self.B.transpose(1, 0),
                                              torch_expm(self.A.transpose(1, 0) * t))

        self.simpson_evals = simpson_evals

        self.WTf = simpson(self.f,
                           0,
                           self.simpson_evals,
                           self.Tf / self.simpson_evals,
                           progress_bar=progress_bar
                           )
        self.x0 = x0s.unsqueeze(-1)  # batch x  n_nodes x 1
        self.xTf = x_infs.unsqueeze(-1)  # batch x n_nodes x 1

        if len(self.x0.shape) == 4 and self.x0.shape[1] == 1:
            self.x0 = self.x0.squeeze(1)

        if len(self.xTf.shape) == 4 and self.xTf.shape[1] == 1:
                self.xTf = self.xTf.squeeze(1)

        if len(self.x0.shape) > 3 or  len(self.xTf.shape) > 3:
            print(self.x0.shape)
            raise ValueError('For this controller, please do not provide state variables '
                             'dimension! Provided '
                             'state tensors must have shape of length 2 for '
                             'accurate '
                             'calculations!')

        self.vTf = self.xTf - torch.matmul(self.expA, self.x0)

        if use_inverse:
            self.WTfm1 = torch.inverse(self.WTf)
            self.ut = self.ut_inverse
        else:
            self.solve = torch.solve(self.vTf, self.WTf)[0]
            self.ut = self.ut_solve

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.ut(t)
    
    def ut_inverse(self, t: torch.Tensor) -> torch.Tensor:
        """
        Control signal calculation for time `t`, this method uses torch.inverse for calculation
        of the inverse to be more computationally accurate, but might require more memory and time.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :return: the control tensor, often of shape: `[b, k]`
        """
        ut = torch.chain_matmul(self.B.transpose(1, 0),
                                torch_expm(self.A.transpose(1, 0) * (self.Tf - t)),
                                self.WTfm1)
        ut = torch.matmul(ut.unsqueeze(0), self.vTf)
        return ut.squeeze(-1)

    def ut_solve(self, t: torch.Tensor) -> torch.Tensor:
        """
        Control signal calculation for time `t`, this method uses torch.solve for approximation
        of the inverse to be more computationally efficient.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :return: the control tensor, often of shape: `[b, k]`
        """
        ut = torch.chain_matmul(self.B.transpose(1, 0),
                                torch_expm(self.A.transpose(1, 0) * (self.Tf - t)))
        ut = torch.matmul(ut, self.solve)
        return ut.squeeze(-1)
