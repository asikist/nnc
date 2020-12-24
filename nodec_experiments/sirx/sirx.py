import torch
from nnc.controllers.base import ControlledDynamics


def get_a_conv(in_channel, out_channel):
    """
    Generate a convolutional layer based on the torch.nn.Conv2d class
    :param in_channel:  the number of input channels
    :param out_channel:  the number of output channels
    :return:
    """
    res = torch.nn.Conv2d(
        in_channel,
        out_channel,
        [1, 3],
        stride=[1, 1],
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
    )
    return res


def neighborhood_mask(association_matrix):
    """
    Based on an interaction/adjacency/association matrix it maps nodes and neighbors by creating
    an appropriate index and mask
    :param association_matrix: The interaction matrix
    :return: a tuple with the mask and index that can convert a state vector of size N, to a ```N x
    d``` matrix where d is the number of max neighborhs.
    """
    max_degree = association_matrix.sum(-1).max().to(torch.long)  # assume symmetric matrix
    mask, inds = association_matrix.sort(-1)
    return mask[:, -max_degree:], inds[:, -max_degree:]


def couple(adjacency, states):
    """
    couples states by using the neighborhood_mask method.
    :param adjacency: the adjacency matrix
    :param states: the state vector
    :return:
    """
    with torch.no_grad():
        mask, inds = neighborhood_mask(adjacency)
        sel_states = states[:, inds]
    coupling = (states.unsqueeze(-1) - sel_states) * mask
    return coupling


def conv_trick(mask, inds, states):
    """
    Couples states by using the result of the neighborhood_mask method.
    :param mask: the neighborhood_mask mask matrix
    :param inds: the sneighborhood_mask index
    :param states: the state vector
    :return:
    """
    sel_states = states[:, inds]
    coupling = sel_states * mask
    return coupling


def flat_to_channels(x, n_nodes, inds, mask):
    """
    takes a flat sirx vector and converts it to a channel vector for convolution.
    :param x:
    :param n_nodes:
    :param inds:
    :param mask:
    :return:
    """
    xinf = x[:, :n_nodes]
    xinf = xinf[:, inds] * mask
    xsusc = x[:, n_nodes:2 * n_nodes]
    xsusc = xsusc[:, inds] * mask
    xrec = x[:, 2 * n_nodes:3 * n_nodes]
    xrec = xrec[:, inds]
    xquar = x[:, 3 * n_nodes:4 * n_nodes]
    xquar = xquar[:, inds] * mask
    x_in = torch.stack([xinf, xsusc, xrec, xquar], dim=1)
    return x_in


class GCNNControl(torch.nn.Module):
    def __init__(self, influence_matrix, driver_matrix, input_preprocessor, budget, in_channels=1,
                 feat_channels=5, message_passes=4):
        """
        Graph neural network for convolution
        :param influence_matrix: the influence or adjacency matrix
        :param driver_matrix: The driver matrix
        :param input_preprocessor: the function that converts the state to a matrix with
        :param budget: The budget for control
        :param in_channels: The number of inpt channels/ state variables
        :param feat_channels: The feature channels in the intermediate architecture
        :param message_passes: The message passes in graph conv.
        """
        super().__init__()
        self.influence_matrix = influence_matrix
        self.driver_matrix = driver_matrix
        self.drivers = torch.where(self.driver_matrix == 1)[0]

        self.n_nodes = self.influence_matrix.shape[0]
        self.n_drivers = self.driver_matrix.shape[0]
        self.max_degree = self.influence_matrix.sum(-1).max().to(torch.long)
        self.message_passes = message_passes
        self.budget = budget

        self.input_preprocessor = input_preprocessor

        j = self.max_degree
        self.modlist = torch.nn.ModuleList()
        in_chans = in_channels

        while j > 2:
            if j - 2 > 2:
                self.modlist.append(get_a_conv(in_chans, feat_channels))
            else:
                self.modlist.append(get_a_conv(in_chans, in_channels))
            in_chans = feat_channels
            j -= 2
        self.modlist.append(torch.nn.AvgPool2d(
            [1, 2],
            stride=[1, 1],
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        ))

    def forward(self, t, x):
        if t is None:
            t = torch.zeros([1])[0]
        z = x
        # input preparation done
        # message pass for 4 turns
        for j in range(self.message_passes):
            i = 0
            z = self.input_preprocessor(
                z)  # go from flat to channels, 4 channels are relevant for SIRX.
            for m in self.modlist:
                # do convolutions until N+1 shape is reached, extending feature space
                z = m(z)
                if i < len(self.modlist) - 1:
                    # after last convolution average pool over features
                    z = torch.relu(z)
                i += 1
            if j < self.message_passes - 1:
                # before last message pass, flatten and preserve batch for following message passes.
                z = z.view(x.shape[0], -1)

        z = z.mean(1).squeeze(1)  # flatten channels and have a shape of: batch x N
        # go from N to M by taking into account the driver matrix related values (driver message inputs)
        u = torch.matmul(self.driver_matrix.unsqueeze(0), z[:, self.drivers, :]).squeeze(-1)

        # softmax to apply budget constraing. Again relevant to the specific SIRX case.
        u = torch.nn.functional.softmax(u, dim=-1) \
            * self.budget \
            * self.driver_matrix.sum(-1)

        return u


class SIRDelta(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,
                 infection_rate,
                 recovery_rate,
                 driver_matrix,
                 k_0,
                 # k=0,
                 ):
        """
        Simplest dynamics of the form: dx/dt = AX + BU
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usuallu "cpu" or "cuda:0"
        """
        super().__init__(['concataned state of shape [4 x n_nodes],' +
                          'with state vars in sequence: infected (I), susceptible (S), recovered (R),' +
                          'containment (X)'])
        self.infection_rate = infection_rate  # beta
        self.recovery_rate = recovery_rate  # gamma
        self.adjacency_matrix = adjacency_matrix  # A
        self.driver_matrix = driver_matrix  # B
        self.label = 'sir'
        self.dtype = self.adjacency_matrix.dtype
        self.device = self.adjacency_matrix.device

        self.n_nodes = self.adjacency_matrix.shape[0]
        self.n_drivers = self.driver_matrix.shape[1]

        self.k_0 = k_0 * self.driver_matrix.sum(-1)
        # self.k = k*self.driver_matrix.sum(-1) #use for closer to Brockman implmentations

    def forward2(self, t, x):
        """
        Alternative forward overlod with control concataned in the end of the state vector. Useful for some tests
        :param x: state vector with control appended
        :param t: time scalar, unused
        """
        return self.forward(x[:, :-self.drivers], t, x[:, -self.drivers:])

    def forward(self, t, x, u=None):
        """
        Dynamics forward overload with seperate control and batching.
        :param x: state vector
        :param t: time scalar, unused
        :param u: control vector
        """
        batch_size = x.shape[0]
        I = x[:, :self.n_nodes]  # I
        S = x[:, self.n_nodes:2 * self.n_nodes]
        R = x[:, 2 * self.n_nodes:3 * self.n_nodes]
        X = x[:, 3 * self.n_nodes:]

        # for sake of generality, treat no control as zeros.
        if u is None:
            u_hat = torch.zeros([batch_size, self.n_drivers], device=x.device, dtype=x.dtype)
        else:
            u_hat = u

            # calc the adjacency matix sum term
        AI = torch.matmul(self.adjacency_matrix.unsqueeze(0), I.unsqueeze(-1)).squeeze(-1)

        # calc terms related to infection 
        I_term = self.infection_rate * S * AI

        # calc terms related to recovery 
        R_term = self.recovery_rate * I

        # calc relevant control terms
        S_control_term = (self.k_0 + u_hat) * S
        I_control_term = (self.k_0 + u_hat) * I

        # calc derivatives
        dI = I_term - R_term - I_control_term
        dR = R_term + S_control_term
        dS = -I_term - S_control_term
        dX = I_control_term

        # stack derivatives to state in received order
        dx = torch.cat([dI, dS, dR, dX], dim=-1)
        return dx
