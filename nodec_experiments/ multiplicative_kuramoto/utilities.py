import torch
from torchdiffeq import odeint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

from copy import deepcopy, copy


def calculate_energy_trajectory(control_signal, total_time, n_control_interactions, sum_axis=-1):
    dt = total_time/n_control_interactions
    return ((control_signal**2)*dt).sum(-1)

def generate_complete_graph(n=225):
    graph = nx.complete_graph(n)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    os.makedirs('../data/', exist_ok=True)
    torch.save(adjacency_matrix, '../data/complete_graph_adjacency.pt')


def generate_erdos_renyi(n=225, p=0.3, seed=1):
    graph = nx.erdos_renyi_graph(n, p, seed=seed)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    os.makedirs('../data/', exist_ok=True)
    torch.save(adjacency_matrix, '../data/erdos_renyi_adjacency.pt')


def generate_square_lattice(side_size=15, seed=1):
    graph = nx.grid_graph([side_size, side_size])
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    os.makedirs('../data/', exist_ok=True)
    torch.save(adjacency_matrix, '../data/square_lattice_adjacency.pt')


def generate_watts_strogatz(n=225, p=0.3, k=5, seed=1):
    graph = nx.watts_strogatz_graph(n, p=p, k=k, seed=seed)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    os.makedirs('../data/', exist_ok=True)
    torch.save(adjacency_matrix, '../data/watts_strogatz_adjacency.pt')


def generate_parameters(total_time,
                         n=225,
                         manual_seed=1
                        ):
    torch.manual_seed(manual_seed)
    parameters = dict()
    parameters['total_time']        = total_time

    # initial angle
    theta_0 = torch.empty([1, 225]).normal_(mean=0, std=0.2)

    # omega
    natural_frequencies = torch.empty([n])\
                               .normal_(mean=0, std=0.2)
    
    parameters['natural_frequencies'] = natural_frequencies
    parameters['theta_0']             = theta_0
    parameters['total_time']          = total_time

    os.makedirs('../data/', exist_ok=True)
    torch.save(parameters, '../data/parameters.pt')


def load_parameters():
    return torch.load('../data/parameters.pt')


def evaluate(dynamics, theta_0, controller, total_time, n_interactions, progress_bar = None):
    timesteps = torch.linspace(0, total_time, n_interactions, device=theta_0.device)
    theta = theta_0
    control_trajectory = [torch.zeros([1, 1], device=theta_0.device)]
    state_trajectory = [theta_0]
    timesteps_range =  range(timesteps.shape[0] - 1)
    if progress_bar is not None:
        timesteps_range = progress_bar(timesteps_range)
    for i in timesteps_range:
        time_start = timesteps[i]
        time_end = timesteps[i + 1]
        current_interval = torch.linspace(time_start, time_end, 2)
        u = controller(time_start, theta)
        controlled_dynamics = lambda t, y: dynamics(t=t,
                                                    x=y,
                                                    u=u)

        theta = odeint(controlled_dynamics,
                       theta,
                       current_interval,
                       method='rk4',
                       options={'step_size': 0.01}
                       )[-1]
        control_trajectory.append(u)
        state_trajectory.append(theta)
    return control_trajectory, state_trajectory


def calculate_critical_coupling_constant(adjacency_matrix, natural_frequencies):
    G = nx.from_numpy_matrix(adjacency_matrix.cpu().detach().numpy())
    laplacian_matrix = np.array(nx.laplacian_matrix(G).todense())
    laplacian_p_inverse = np.linalg.pinv(laplacian_matrix)
    inner_prod_lapl_nat = laplacian_p_inverse @ natural_frequencies.detach().numpy()
    coupling_constant = torch.tensor([np.abs(inner_prod_lapl_nat[np.newaxis, :] -\
                       inner_prod_lapl_nat[:, np.newaxis]).max()*G.number_of_nodes()]).float()
    coupling_constant =  coupling_constant
    return coupling_constant



def comparison_plot(  nodec_line, 
                agd_line, 
                x_data,
                y_label, 
                x_label = '$t$',
                figsize=(3, 3), 
                dpi=120, 
                facecolor='w',          
                legend_loc='lower center'
             ):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    ax1 = fig.add_subplot(111)
    ax1.plot(x_data, agd_line, label='Adjoint Gradient', color='#fdb863')

    ax1.plot(x_data, nodec_line,  label='NODEC', color='#5e3c99')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    leg = ax1.legend(loc=legend_loc)
    return fig 

def state_plot( state_line, 
                x_data,
                y_label, 
                x_label = '$t$',
                figsize=(3, 3), 
                dpi=120, 
                facecolor='w',          
                legend_loc='lower center'
             ):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    ax1 = fig.add_subplot(111)
    ax1.plot(x_data, state_line)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    #leg = ax1.legend(loc=legend_loc)
    return fig 





def state_heatmap_plot(data, vmin=0, vmax=1):    
    fig, ax = plt.subplots()
    cmap = copy(plt.cm.plasma)
    cmap.set_over('white')
    cmap.set_under('black')
    im = ax.imshow(data, vmin=-vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(im, extend='both')
    im.axes.xaxis.set_visible(False)
    im.axes.yaxis.set_visible(False)
    return fig, ax

if __name__ == '__main__':
    total_time = 3.0
    generate_parameters(total_time)
    generate_complete_graph()
    generate_square_lattice()
    generate_erdos_renyi()
    generate_watts_strogatz()



    