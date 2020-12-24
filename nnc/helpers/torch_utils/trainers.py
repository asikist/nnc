import torch.optim
from torchdiffeq import odeint
from torch.optim.optimizer import Optimizer
import numpy as np
from copy import deepcopy
import sys
from nnc.helpers.torch_utils.numerics import faster_adj_odeint


class NODECTrainer:
    def __init__(self,
                 nodec_dynamics,
                 x0,
                 x_target,
                 total_time,
                 obj_function=None,
                 optimizer_class: Optimizer = torch.optim.Adam,
                 optimizer_params=None,
                 ode_solver_kwargs=None,
                 logger=None,
                 closure=None,
                 use_adjoint=False,
                 log_freq=5,
                 print_freq=10
                 ):
        """
        A trainer for Nodec that uses adaptive learning rate and reloads and preserves the best
        model when encountering numerical instabilities.
        :param nodec_dynamics: The nodec dynamics to train.
        Please see ` nnc.controllers.neural_network.nnc_controllers.NNCDynamics`
        :param x0: the initial state
        :param x_target: the target state
        :param total_time: the total time
        :param obj_function: the objective function to use, often written
        as `lambda x, x_target: scalar`.
        Should allow gradient propagation.
        :param optimizer_class: The `torch.optim.Class`
        :param optimizer_params: The keyworded parameters to reinitialize the optimizer after
        numericall instabilities.
        :param ode_solver_kwargs: The keyworded parameters to initialize the odesolver.
        :param logger: The `nnc.helper.torch_utils.evaluators` object to use for logging while
        trainnig.
        :param closure: The optimizer closure that returns the loss.
        :param use_adjoint: Whether to use adjoint in ode solver and backprop for memory efficiency.
        :param log_freq: The frquency at which a logger is called in epochs.
        :param print_freq: The frequency at which the stdout is updated with training info.
        """
        self.nodec_dynamics = nodec_dynamics
        self.x0 = x0
        self.x_target = x_target
        self.total_time = total_time

        if obj_function is None:
            obj_function = torch.nn.MSELoss()

        self.obj_function = obj_function
        self.optimizer_class = optimizer_class

        if optimizer_params is None:
            optimizer_params = {'lr': 0.1}
        self.optimizer_params = optimizer_params

        if ode_solver_kwargs is None:
            ode_solver_kwargs = dict(method='dopri5')
        if not use_adjoint:
            ode_solver = lambda dynamics, x, t_linspace: \
                odeint(dynamics, x, t_linspace, **ode_solver_kwargs)
        else:
            ode_solver = lambda dynamics, x, t_linspace: \
                faster_adj_odeint(dynamics, x, t_linspace, **ode_solver_kwargs)

        self.ode_solver = ode_solver
        self.t_linspace = torch.linspace(0,
                                         self.total_time,
                                         2,
                                         device=self.x0.device,
                                         dtype=self.x0.dtype
                                         )
        self.optimizer = self._get_optimizer()
        self.initial_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.current_loss = np.infty
        self.best_loss = self.current_loss
        self.best_model_params = deepcopy(self.nodec_dynamics.state_dict())
        self.previous_loss = float('inf')
        self.logger = logger
        self.loss_variance_tolerance = 10
        self.log_freq = log_freq
        self.print_freq = print_freq

        if closure is not None:
            self.closure = closure
        else:
            self.closure = self._closure

    def _get_optimizer(self):
        """
        Creates a new optimizer.
        :return:
        """
        return self.optimizer_class(self.nodec_dynamics.parameters(),
                                    **self.optimizer_params
                                    )

    def _closure(self):
        """
        default closure.
        :return:
        """
        x_reached = self.ode_solver(self.nodec_dynamics, self.x0, self.t_linspace)[-1]
        loss = self.obj_function(x_reached, self.x_target)
        loss.backward()
        loss_val = loss.item()
        self.current_loss = loss_val
        if not self.current_loss < self.loss_variance_tolerance * self.previous_loss:
            raise ValueError('Potential exploding loss value encountered')
        if self.current_loss < self.best_loss and np.isfinite(self.current_loss):
            self.best_loss = self.current_loss
            self.best_model_params = deepcopy(self.nodec_dynamics.state_dict())
        self.previous_loss = self.current_loss

        return loss.item()

    def train_best(self,
                   epochs=50,
                   loss_variance_tolerance=10,
                   lr_deceleration_rate=0.5,
                   lr_acceleration_rate=0.001,
                   verbose=False,
                   progress_bar=None
                   ):
        """
        Train the stored NODEC and return the best model.
        :param epochs: Number of epochs to train.
        :param loss_variance_tolerance: The upper threshold to the new_loss/previous_loss to
        determine an instability.
        :param lr_deceleration_rate: The ratio at which the learning rate decreases at
        instabilities.
        :param lr_acceleration_rate: the extra ratio (added to 1) at which learning rate
        increases over time.
        :param verbose: Whetehr to print intermediate training results.
        :param progress_bar: A tqdm progress bar.
        :return:
        """
        exception_text = 'None'
        epoch_gen = range(epochs)
        if progress_bar is not None:
            epoch_gen = progress_bar(epoch_gen)
        for epoch in epoch_gen:
            if verbose and (epoch == epochs - 1 or epoch % self.print_freq == 0):
                sys.stdout.write("Epoch: %d , Current Loss: %f, Best Loss: %f, Current lr: %f, "
                                 "last instability content: %s \r"
                                 % (
                                     epoch,
                                     self.current_loss,
                                     self.best_loss,
                                     self.optimizer.state_dict()['param_groups'][0]['lr'],
                                     str(exception_text)
                                 )
                                 )
                sys.stdout.flush()
            try:
                self.loss_variance_tolerance = loss_variance_tolerance
                self.optimizer.zero_grad()
                self.optimizer.step(self.closure)
                if 'cuda' in str(self.x0.device):
                    torch.cuda.empty_cache()
                if self.logger is not None and (epoch == epochs - 1 or epoch % self.log_freq == 0):
                    with torch.no_grad():
                        eval_result = self.logger.evaluate(self.nodec_dynamics.underlying_dynamics,
                                                           self.nodec_dynamics.nnc,
                                                           self.x0,
                                                           self.t_linspace[-1],
                                                           epoch
                                                           )
                        self.logger.write_to_file(eval_result,
                                                  other_metada={
                                                      'lr': self.optimizer_params['lr'],
                                                      'optimizer': self.optimizer.__class__.__name__,
                                                      'dynamics': self.nodec_dynamics.underlying_dynamics.__class__.__name__,
                                                      'neural_net': self.nodec_dynamics.nnc.neural_net.__class__.__name__
                                                  })
            except (AssertionError, ValueError) as error:
                # print('Numerical instability encountered at epoch: ' + str(epoch))
                exception_text = str(error) + ' at epoch ' + str(epoch)
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.optimizer_params['lr'] = lr * lr_deceleration_rate
                self.optimizer = self._get_optimizer()
                self.nodec_dynamics.load_state_dict(deepcopy(self.best_model_params))
            if lr_acceleration_rate > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = (1 + lr_acceleration_rate) * param_group['lr']
        self.nodec_dynamics.load_state_dict(deepcopy(self.best_model_params))
        return self.nodec_dynamics
