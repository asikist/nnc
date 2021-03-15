import logging
import numpy as np
import torch
from torchdiffeq import odeint
from nnc.controllers.base import BaseController


class AdjointGD(BaseController):

    def __init__(self,
                 forward_dynamics,
                 backward_dynamics,
                 theta_0,
                 n_timesteps,
                 total_time,
                 learning_rate=1,
                 beta=10 ** -7,
                 iterations=10,
                 control_change_tolerance=10 ** -5,
                 progress_bar=None,
                 ode_int_kwargs=None
                 ):
        super().__init__()
        self.forward_dynamics = forward_dynamics
        self.backward_dynamics = backward_dynamics
        self.n_nodes = self.forward_dynamics.adjacency_matrix.shape[-1]
        self.device = self.forward_dynamics.adjacency_matrix.device
        self.n_timesteps = n_timesteps
        self.timesteps = torch.linspace(0,
                                        total_time,
                                        n_timesteps,
                                        device=self.device
                                        )
        self.total_time = total_time
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = iterations
        self.control_change_tolerance = control_change_tolerance
        self.progress_bar = progress_bar
        self.ode_int_kwargs = ode_int_kwargs
        self.u_baseline = 1+ torch.rand([theta_0.shape[0],
                                       self.n_timesteps, 1],
                                      device=self.device,
                                      requires_grad=False
                                      )
        self.theta_0 = theta_0
        self._learn(theta_0)

    def _dtheta(self, t, theta):
        u = self.forward(t)
        return self.forward_dynamics(
            t,
            u=u,
            x=theta
        )

    def _dp(self, t, p, thetas):
        u = self.forward(t).detach()
        theta = thetas[torch.argmin(torch.abs(self.timesteps - t))]
        return - self.backward_dynamics(
            t=t,
            x=torch.stack([p, theta], 1),
            u=u
        )

    def _learn(self, theta_0):
        control_value_change = np.infty
        iteration = 0

        adjacency_matrix = self.forward_dynamics.adjacency_matrix
        coupling_constant = self.forward_dynamics.coupling_constant
        while control_value_change > self.control_change_tolerance and iteration < self.iterations:
            iteration = iteration + 1
            try:
                #forward integration
                thetas = odeint(self._dtheta,
                                theta_0.detach(),
                                self.timesteps.detach(),
                                **self.ode_int_kwargs
                                )
                theta_total_time = thetas[-1]

                # adjoint state at T: total_time
                sin_theta_total_time = torch.sin(2 * theta_total_time)
                cos_theta_total_time = torch.cos(2 * theta_total_time)
                sin_product_p = sin_theta_total_time * (
                    (adjacency_matrix @ cos_theta_total_time.unsqueeze(-1)).squeeze(-1)) -\
                                cos_theta_total_time * (
                     (adjacency_matrix @ sin_theta_total_time.unsqueeze(-1)).squeeze(-1)
                    )
                p_total_time = 1 / 2 * sin_product_p
                
                # backward integration
                all_p = odeint(
                    lambda t, y: self._dp(t, y, thetas),
                    p_total_time,
                    self.timesteps,
                    **self.ode_int_kwargs
                ) 
                sin_thetas = torch.sin(thetas).unsqueeze(-1)
                cos_thetas = torch.cos(thetas).unsqueeze(-1)
                sin_product = cos_thetas * (
                              adjacency_matrix.unsqueeze(0).unsqueeze(0) @ sin_thetas) - \
                              sin_thetas * (adjacency_matrix.unsqueeze(0).unsqueeze(0) @ cos_thetas)
                u_update_term = (all_p.unsqueeze(-2) @ sin_product).squeeze(-1).squeeze(-1)
                control_value_change = self.u_baseline.detach()
                self.u_baseline = self.u_baseline - \
                                  self.learning_rate * (
                                          self.beta * self.u_baseline + 
                                          (coupling_constant / self.n_nodes) *
                                          u_update_term)
                control_value_change = torch.mean((control_value_change - self.u_baseline) ** 2)
                # loss = (sinTprod**2).mean()
                loss = ((self.forward_dynamics.laplacian_matrix
                          @ theta_total_time.unsqueeze(-1)) ** 2).sum()
               
                logging.info('GD step loss: ' + str(loss.item()))

            except AssertionError:
                logging.info("ODESolver encountered numerical instability, gradient descent stops!")
                return

    def forward(self, t: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        # we perform a midpoint interpolation between controls
        t_diff = t - self.timesteps
        t_prev_diff = t_diff.clone()
        t_prev_diff[t_prev_diff < 0] = np.infty
        t_prev_idx = torch.argmin(t_prev_diff)
        u_prev = self.u_baseline[:, t_prev_idx]
        t_next_diff = t_diff
        t_next_diff[t_next_diff >= 0] = -np.infty
        t_next_idx = torch.argmax(t_next_diff)

        t_prev = self.timesteps[t_prev_idx]
        t_next = self.timesteps[t_next_idx]
        u_next = self.u_baseline[:, t_next_idx]
        time_weight = (t-t_prev)/(t_next-t_prev)
        u = u_prev*(1-time_weight)*u_prev + u_next*(time_weight)
        return u_prev
        # return self.u_baseline[:, torch.argmin(torch.abs(self.timesteps - t)), :]
