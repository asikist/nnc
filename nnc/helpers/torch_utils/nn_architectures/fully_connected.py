import torch
from torch.nn import Linear
from nnc.helpers.torch_utils.indexing import multi_unsqueeze

class StackedDenseTimeControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of linear systems
    """

    def __init__(self, n_nodes, n_drivers, n_hidden=1, hidden_size=10,
                 activation=torch.nn.functional.elu, use_bias=True):
        super().__init__()
        self.input_layer = torch.nn.Linear(1, hidden_size, bias=use_bias)
        hidden_layers = []
        for i in range(n_hidden):
            hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size, bias=use_bias))
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.output_layer = torch.nn.Linear(hidden_size, n_drivers, bias=use_bias)
        self.activation = activation

    def forward(self, t, x):
        """
        :param t: A scalar or a batch with scalars
        :param x: input_states for all nodes
        :return:
        """
        t = t
        tshape = len(t.shape)
        if tshape < 2:
            t = multi_unsqueeze(t, 2-tshape)
        u = self.input_layer(t)
        u = self.activation(u)
        for module in self.hidden_layers:
            u = module(u)
            u = self.activation(u)
        u = self.output_layer(u)
        return u


class StackedDenseFeedbackControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of  systems with feedback
    """

    def __init__(self, n_nodes, n_drivers, n_hidden=1, hidden_size=10,
                 activation=torch.nn.functional.elu, use_bias=True):
        super().__init__()
        self.input_layer = torch.nn.Linear(n_nodes, hidden_size, bias=use_bias)
        hidden_layers = []
        for i in range(n_hidden):
            hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size, bias=use_bias))
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.output_layer = torch.nn.Linear(hidden_size, n_drivers, bias=use_bias)
        self.activation = activation

    def forward(self, t, x):
        """
        :param t: A scalar or a batch with scalars
        :param x: input_states for all nodes
        :return:
        """
        u = self.input_layer(x)
        u = self.activation(u)
        for module in self.hidden_layers:
            u = module(u)
            u = self.activation(u)
        u = self.output_layer(u)
        return u
