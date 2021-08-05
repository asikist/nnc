import torch
from torchdiffeq import odeint
import numpy as np
from nnc.controllers.neural_network.nnc_controllers import NNCDynamics
import logging




from copy import deepcopy
import signal
from contextlib import contextmanager


class EluTimeControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of linear systems
    """

    def __init__(self, hidden_layer_sizes=(15,)):
        super().__init__()

        total_sizes = [1] + hidden_layer_sizes + [1]
        module_list = []
        for i in range(len(total_sizes) - 1):
            module_list.append(torch.nn.Linear(total_sizes[i],
                                               total_sizes[i + 1]))
        self.layers = torch.nn.ModuleList(module_list)

    def forward(self, t, x):
        """
        :param t: A scalar or a batch with scalars
        :param x: input_states for all nodes
        :return:
        """
        t = t.detach()
        # check for batch sizes and if t is scalar:
        if len(t.shape) in {0, 1}:
            if x is not None and len(list(x.shape)) > 1:
                t = t.repeat(x.shape[0], 1)
            else:
                t = t.unsqueeze(0)
        u = self.layers[0](t)
        for i in range(1, len(self.layers) - 1):
            u = self.layers[i](u)
            u = torch.nn.functional.elu(u)

        u = self.layers[-1](u)
        return u


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def calculate_loss(A, x):
    x_r = x.unsqueeze(-2) # repeat elements in rows
    x_c = x.unsqueeze(-1) # repeat elements in columns
    diff = x_r-x_c # create the difference of x_j - x_i in each row element
    sin_sqr_diff   =  torch.sin(diff)**2
    adjacency_mult = A*sin_sqr_diff
    loss = adjacency_mult.sum() # sum over i and j
    return loss


class TrainingAlgorithm:
    def __init__(self, net, dynamics):
        self.nnc_dyn = NNCDynamics(dynamics, net)
        self.neural_net = net
        self.adjacency_matrix = self.nnc_dyn.underlying_dynamics.adjacency_matrix
        self.optimizer = None

    def train(self, theta_0, total_time, epochs=9, lr=1):
        """
        Trains a neural network control via the NODEC framework.
        :param theta_0: The initial state
        :param total_time: The total time for the control trajectory
        :param epochs: The number of epochs to train
        :param lr: The initial learning rate
        :return: The best performing neural network
        """

        # 1. Prepare local parameters such as initial state and the time linspace
        x0 = theta_0.detach()
        if len(theta_0.shape) == 1:
            x0 = x0.unsqueeze(1)
        t = torch.linspace(0, total_time, 2)
        self.optimizer = torch.optim.SGD(self.nnc_dyn.parameters(), lr=lr)
        # optimizer = torch.optim.LBFGS(nnc_dyn.parameters(), lr=0.00001)

        # 2. initialize a mutable container for the best neural network and loss.
        best_model = [deepcopy(self.neural_net)]
        best_loss = [np.infty]

        for i in range(epochs):
            # Do not allow gradient accumulation through epochs
            self.optimizer.zero_grad()
            try:
                def closure():
                    """
                    An optimization step for learning
                    :return:
                    """
                    x = x0.detach()

                    # A. integrate the dynamics and the neural network control
                    x_total_time = odeint(self.nnc_dyn, x, t, method='dopri5')[-1, :]

                    # B. Calculate the loss as the inner product between the laplacian
                    #  and the reached state. (state at total time)
                    # loss = ((self.graph_laplacian @\
                    #          x_total_time.unsqueeze(-1)) ** 2
                    #         ).sum()


                    loss = calculate_loss(self.adjacency_matrix,
                                          x_total_time.squeeze(-1)
                                          )
                    logging.info('Current loss is: ' + str(loss.item()))

                    if loss.detach().cpu().item() < best_loss[0]:
                        # C.a If the loss is better than best loss, then
                        #  preserve a copy of the parameters and update the best loss
                        best_loss[0] = loss.detach().cpu().item()
                        best_model[0] = deepcopy(self.nnc_dyn.nnc.neural_net)
                    elif loss.detach().cpu().item() > 10 * best_loss[0]:
                        # C.b If the loss is 10 times higher than the best observed loss,
                        #  assume a numerical instability occurred and raise an error.
                        raise ValueError('Numerical Instability')

                    # D. ack-propagate the loss gradient
                    loss.backward()
                    return loss

                with time_limit(50):
                    # 3. Forward and backpropagate within 50 seconds
                    #  3.a if the operation takes more than 50 seconds, assume
                    #   numerical instability (solver becomes slow due to stiffness)
                    #   occurs and raise an error.
                    logging.info('Training epoch: ' + str(i))
                    self.optimizer.step(closure)

            except Exception as e:
                # 4. When an exception related to potential numerical instabilities occurs
                #  reset parameter values to the best performing model observed so far
                #  and reinitialize the optimizer with half learning rate.
                self.nnc_dyn.nnc.neural_net = best_model[0]
                neural_net = best_model[0]
                lr = lr / 2
                logging.info('Numerical Instability at training epoch: ' + str(i) + '.\n' +
                                ' Resuming parameters to best perfroming model. ' +
                                ' Re-initializing the SGD optimizer with lr: ' + str(lr)
                                )
                logging.info(e)
                self.optimizer = torch.optim.SGD(neural_net.parameters(), lr=lr)
        return best_model[0]
