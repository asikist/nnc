import gym
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint


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


class SIRXEnv(gym.Env):
    def __init__(self, env_config):
        """
        Gym environment for SIRX, in env config one can find everything required for the dynamics.
        Please check jupyter notebooks for more
        :param env_config:
        """

        self.sirx = env_config['sirx']
        target_nodes = env_config['target_nodes']

        self.n_nodes = self.sirx.adjacency_matrix.shape[1]
        self.n_drivers = self.sirx.driver_matrix.shape[1]
        self.target_nodes = [target_nodes, np.arange(self.n_nodes)][target_nodes is None]

        self.state_size = 4 * self.n_nodes
        self.action_size = self.n_drivers
        self.x_init = env_config['x0']
        self.state = self.x_init[0].detach().cpu().numpy()

        self.dt = env_config['dt']
        self.total_time = env_config['T']
        self.time_steps = torch.linspace(0, self.total_time, int(self.total_time // self.dt),
                                         device=self.x_init.device)
        self.budget = env_config['budget']

        self.action_space = gym.spaces.Box(low=-100, high=100, shape=[
            self.n_drivers])  # these are logits so any closed space in R would do
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[self.state_size])

        self.spaces = [self.action_space, self.observation_space]

        self.reward_range = (-np.inf, 0)

        self.ode_solve_method = env_config['ode_solve_method']

        if "reward_type" in env_config.keys():
            self.reward_type = env_config['reward_type']
        else:
            self.reward_type = 'minus_l2'
        self.reset()

    def step(self, actions, is_logit=True):
        # action preprocessing for compatibility with batched versions of dynamics and ODE integrators
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions).to(device=self.x_init.device, dtype=torch.float)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[-1] == self.n_drivers:
            # go from N to M by taking into account the driver matrix related values (driver message inputs)
            actions = torch.matmul(self.sirx.driver_matrix.unsqueeze(0),
                                   actions.unsqueeze(-1)).squeeze(-1)

        # provided actions are considered to be logits for the softmax
        # which does budget assignment and produces final u
        if is_logit:
            # softmax to apply budget constraing. Again relevant to the specific SIRX case.
            u = torch.nn.functional.softmax(actions, dim=-1) \
                * self.budget \
                * self.sirx.driver_matrix.sum(-1)
        else:
            u = actions

        # time for numerical ode solve
        integr_time = self.time_steps[self.c_timestep: self.c_timestep + 2]
        # print(integr_time)
        deriv = lambda t, x: self.sirx(t, x, u=u)
        if self.ode_solve_method == 'rk4':
            next_state = odeint(deriv, self.state, integr_time, method='rk4')[-1]
        else:  # e.g. dopri5, adams
            next_state = odeint(deriv, self.state, integr_time, method=self.ode_solve_method)[-1]

        # state update
        self.state = next_state

        # determining when done
        done = self.c_timestep == self.time_steps.shape[0] - 1

        # reward calculation
        # for now old skool type without oop and general coding stuff
        reward = None
        if self.reward_type == "minus_l2":
            reward = self.minus_l2(next_state)
        elif self.reward_type == "sparse_end":
            reward = self.sparse_end(next_state)
        elif self.reward_type == 'sum_to_max':
            reward = self.sum_to_max(next_state)
        else:
            raise ValueError(
                "Wrong reward type provided! Please choose either \'minus_l2\' or \'sparse_end'\''")
        self.c_timestep += 1
        return self.state.detach().cpu().numpy()[0], reward.detach().cpu().item(), done, {}

    def minus_l2(self, state):

        # motivation is that the return of this reward will calculate the integral till t*

        return -((state[:, self.target_nodes] ** 2) * self.dt).mean()

    def sparse_end(self, state):
        if self.c_timestep == 0:
            self.max_inf = torch.tensor([0.0]).to(device=state.device)
        mean_inf = state[:, self.target_nodes].mean()
        self.max_inf = [self.max_inf, mean_inf][mean_inf > self.max_inf]
        # motivation is that this reward has the min minus inf rate
        if self.c_timestep == self.time_steps.shape[0] - 1:
            return -(self.max_inf) ** 2
        else:
            return torch.tensor([0.0]).to(device=state.device)

    def sum_to_max(self, state):
        if self.c_timestep == 0:
            self.max_inf = torch.tensor([0.0]).to(device=state.device)
        mean_inf = state[:, self.target_nodes].mean()
        prev_inf = self.max_inf + 0  # how to copy in a lazy way
        self.max_inf = [self.max_inf, mean_inf][mean_inf > self.max_inf]
        # motivation is that this reward has the min minus inf rate
        if prev_inf != self.max_inf:
            return -(self.max_inf) ** 2 + prev_inf ** 2
        else:
            return torch.tensor([0.0]).to(device=state.device)

    def reset(self):
        self.c_timestep = 0
        self.state = self.x_init
        return self.x_init[0].detach().cpu().numpy()

    def render(self):
        raise ValueError("Not implemented yet.")

    def close(self):
        # raise ValueError("Not implemented yet.")
        pass

    def seed(self, seed):
        pass
        # raise ValueError("Not implemented yet.")


class RLGCNN(torch.nn.Module):
    'RL graph conv for RL'

    def __init__(self, adjacency_matrix, driver_matrix, input_preprocessor,
                 in_channels=1, feat_channels=5, message_passes=4):
        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.driver_matrix = driver_matrix
        self.drivers = torch.where(self.driver_matrix == 1)[0]

        self.n_nodes = self.adjacency_matrix.shape[0]
        self.n_drivers = self.driver_matrix.shape[0]
        self.max_degree = self.adjacency_matrix.sum(-1).max().to(torch.long)
        self.message_passes = message_passes

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

    def forward(self, x, t=torch.zeros([1])[0]):
        z = x
        # input preparation done
        # message pass for 4 turns
        for j in range(self.message_passes):
            i = 0
            z = self.input_preprocessor(z)  # go from flat to channels, relevant for SIRX.
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
        u = z[:, self.drivers, :].squeeze(-1)

        # sotmax would be here, but instead we do the logits now    
        return u


class Actor(nn.Module):
    """
    Simple critic network based on tianshou implementation
    """
    def __init__(self, model, device='cuda:0'):
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, s, **kwargs):
        s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, model, action_shape, device='cpu'):
        super().__init__()
        self.device = device
        self.model = model
        self.mu = nn.Linear(np.prod(action_shape), np.prod(action_shape))
        self.sigma = nn.Linear(np.prod(action_shape), np.prod(action_shape))

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = torch.tanh(self.mu(logits))
        sigma = torch.exp(self.sigma(logits))
        return (mu, sigma), None


class Critic(nn.Module):
    """
    Simple critic network based on tianshou implementation
    """
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(state_shape + action_shape, state_shape),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(state_shape, state_shape), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(state_shape, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if a is not None and not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)

        if a is None:
            logits = self.model(s)
        else:
            a = a.view(batch, -1)
            logits = self.model(torch.cat([s, a], dim=1))
        return logits


def transform_u(logits, driver_matrix, budget):
    """
    A function that transforms RL logits to valid controls.
    :param logits: RL action logits
    :param driver_matrix: driver matrix for control selection
    :budget: total budget available
    """
    logits = torch.matmul(driver_matrix.unsqueeze(0), logits.unsqueeze(-1)).squeeze(-1)
    u = torch.nn.functional.softmax(logits, dim=-1) \
        * budget \
        * driver_matrix.sum(-1)
    return u
