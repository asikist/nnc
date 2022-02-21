import torch
import pandas as pd
import numpy as np
from torchdiffeq import odeint
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data operations

class EluTimeControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of linear systems
    """
    def __init__(self, n_nodes, n_drivers):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_nodes+4)
        self.linear0 = torch.nn.Linear(n_nodes+4,n_nodes+4)
        self.linear1 = torch.nn.Linear(n_nodes+4, n_nodes+4)
        self.linear_final = torch.nn.Linear(n_nodes+4, n_drivers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Control calculation via a fully connected NN.
        :param t: A scalar or a batch with scalars, shape: `[b, 1]` or '[1]'
        :param x: input_states for all nodes, shape `[b, m, n_nodes]`
        :return:
        """
        # sanity check to make sure we don't propagate through time
        t = t.detach() # we do not want to learn time :)
        # check for batch size and if t is scalar:
        if len(t.shape)  in {0 , 1} :
            if x is not None and len(list(x.shape)) > 1:
                t = t.repeat(x.shape[0], 1)
            else:
                # add single sample dimension if t is scalar or single dimension tensor
                # scalars are expected to have 0 dims, if i remember right?
                t = t.unsqueeze(0)
        u = self.linear(t)
        u = torch.nn.functional.elu(u)
        u = self.linear0(u)
        u = torch.nn.functional.elu(u)
        u = self.linear_final(u)
        return u


def todf(trajectory, lr=None):
    """
    Converts a numpy tensor of two node single state variable trajectories to a dataframe for
    easier visualization.
    :param trajectory:
    :param lr:
    :return: the dataframe with all trajectories and metadata
    """
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory)
    else:
        trajectory = trajectory.detach()
    df_all_trajectories = []
    n_timesteps = trajectory.shape[-2]
    for i in range(trajectory.shape[0]):
        sample_id = i
        timestep_ids = np.arange(trajectory.shape[-2])
        df_trajectory = pd.DataFrame(dict(
            x1=trajectory[sample_id, :, 0],
            x2=trajectory[sample_id, :, 1],
            u=trajectory[sample_id, :, 2],
            sample_id=i,
            time=timestep_ids,
        ))
        df_trajectory['reached'] = df_trajectory['time'] == n_timesteps - 1
        if lr is not None:
            df_trajectory['lr'] = lr
        df_all_trajectories.append(df_trajectory)
    df_all_trajectories = pd.concat(df_all_trajectories)
    return df_all_trajectories


def evaluate_trajectory(dynamics, controller, x0, total_time, n_timesteps, method='rk4',
                        options=None):
    all_controls = []
    all_control_times = []
    all_timesteps = torch.linspace(0, total_time, n_timesteps)

    def apply_control(t, x):
        u = controller(t, x)
        #print(u.shape)
        all_control_times.append(t)
        all_controls.append(u)
        dx = dynamics(t=t, x=x, u=u)
        return dx

    trajectory = odeint(apply_control,
                        x0,
                        all_timesteps,
                        method=method,
                        options=options
                        )  # timesteps x n_nodes

    all_controls = torch.stack(all_controls, 0)  # timesteps x n_nodes
    all_control_times = torch.stack(all_control_times)  # timesteps x 1


    # align ode_solver control timesteps with requested timesteps
    _, relevant_time_index = closest_previous_time(all_timesteps, all_control_times)
    relevant_controls = all_controls[relevant_time_index, :]
    return torch.cat([trajectory, relevant_controls], -1)


def closest_previous_time(requested_times, solver_times):
    requested_times = requested_times.unsqueeze(1)
    solver_times = solver_times.unsqueeze(0)
    difft = (requested_times - solver_times)
    difft = difft
    difft[difft < 0] = np.infty
    time_index = difft.argmin(1).flatten()
    return solver_times.squeeze()[time_index], time_index


# Plotting
def plot_vector_field(ld, n_points, min_x1, max_x1, min_x2, max_x2, use_streamline=True):
    x1_space = torch.linspace(min_x1, max_x1, n_points).detach().numpy()
    x2_space = torch.linspace(min_x2, max_x2, n_points).detach().numpy()
    mesh = np.meshgrid(x1_space, x2_space)
    assert np.all(mesh[0].flatten().reshape([n_points, n_points]) == mesh[0])
    x1_line = torch.tensor(mesh[0].flatten())
    x2_line = torch.tensor(mesh[1].flatten())
    xxx = torch.stack([x1_line, x2_line], 1)
    dx = ld(t=None, x=xxx, u=None ) # calculate gradient for all mech grid pairs
    dx1 = dx[:, 0].detach().numpy().reshape([n_points,n_points])
    dx2 = dx[:, 1].detach().numpy().reshape([n_points,n_points])
    if not use_streamline:
          return ff.create_quiver(mesh[0], mesh[1], u=dx1, v=dx2,
                           scale=.035,
                           arrow_scale=.3,
                           name='Vector Field',
                           line_width=1.2
                                 )
    else:
        return ff.create_streamline(x1_space,
                                     x2_space,
                                     u=dx1,
                                     v=dx2,
                                     arrow_scale=0.09,
                                    density = 0.4
                                    )

axis_temp = dict(showline=True,
                 zeroline=False,
                 linewidth=1,
                 linecolor='black',
                 mirror=True,
                 showgrid=False,
                 )
base_temp = dict(
                template = 'plotly_white',
                font = dict(family = 'Times New Roman')
            )
def plot_base_template(x1_min, x1_max, x2_min, x2_max):
    return dict(
                template = 'plotly_white',
                xaxis = dict(showline=True,
                             zeroline=False,
                             linewidth=1,
                             linecolor='black',
                             mirror=True,
                             showgrid=False,
                             title=r"$x_1\text (Controlled)$",
                             range = [x1_min, x1_max]
                            ),
                yaxis = dict(showline=True,
                             linewidth=1,
                             zeroline=False,
                             linecolor='black',
                             mirror=True,
                             showgrid=False,
                             title=r"$x_2$",
                             range = [x2_min, x2_max]
                            ),
                font = dict(family = 'Times New Roman')
            )

def plot_trajectory_comparison(linear_dynamics,
                               x0,
                               x_target,
                               nnc_trajectory,
                               oc_trajectory,
                               x1_min,
                               x1_max,
                               x2_min,
                               x2_max
                               ):
    vector_field = plot_vector_field(
        linear_dynamics,
        n_points=40,
        min_x1=x1_min,
        max_x1=x1_max,
        min_x2=x2_min,
        max_x2=x2_max,
        use_streamline=True
    ).data[0]
    vector_field.line.color = 'rgba(128, 135, 130, 0.5)'
    vector_field.name = 'vector field'
    x_start = go.Scatter(x=x0[0, 0].detach().cpu().numpy(),
                         y=x0[0, 1].detach().cpu().numpy(),
                         name=r'$x_0$',
                         marker=dict(symbol='cross',
                                     color='#9bc59d',
                                     size=10
                                     ),
                         showlegend=True
                         )
    x_goal = go.Scatter(x=x_target[0, 0].detach().cpu().numpy(),
                        y=x_target[0, 1].detach().cpu().numpy(),
                        name=r'$x^*$',
                        mode='markers',
                        marker=dict(symbol='star',
                                    color='#b02e0c',
                                    size=10
                                    ),
                        showlegend=True
                        )

    oc_trajectory_line = px.line(oc_trajectory, x='x1', y='x2').data[0]
    oc_trajectory_line.line.color = '#271f30'
    oc_trajectory_line.line.dash = 'dot'
    oc_trajectory_line.name = 'OC'
    oc_trajectory_line.showlegend = True
    nnc_trajectory_line = px.line(nnc_trajectory, x='x1', y='x2').data[0]
    nnc_trajectory_line.line.color = '#ff9f00'
    nnc_trajectory_line.line.dash = 'dot'
    nnc_trajectory_line.name = 'NODEC'
    nnc_trajectory_line.showlegend = True
    fig = go.Figure([vector_field, x_start, x_goal, oc_trajectory_line, nnc_trajectory_line])
    fig = fig.update_layout(plot_base_template(x1_min, x1_max, x2_min, x2_max))
    return fig

def compare_trajectories(linear_dynamics,
                         oc_baseline,
                         nnc,
                         x0,
                         x_target,
                         T,
                         x1_min,
                         x1_max,
                         x2_min,
                         x2_max,
                         n_points=200,
                         ):
    trajectory = evaluate_trajectory(linear_dynamics,
                                     oc_baseline,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    oc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    trajectory = evaluate_trajectory(linear_dynamics,
                                     nnc,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    nnc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    fig_trajectories = plot_trajectory_comparison(linear_dynamics,
                                                  x0,
                                                  x_target,
                                                  nnc_trajectory,
                                                  oc_trajectory,
                                                  x1_min,
                                                  x1_max,
                                                  x2_min,
                                                  x2_max
                                                  )

    energy_nnc = ((nnc_trajectory['u']**2)*T/n_points).cumsum()
    energy_oc = ((oc_trajectory['u'] ** 2) * T / n_points).cumsum()
    time = nnc_trajectory.index*T / n_points

    ocen=go.Scatter(x=time, y=energy_oc, name='OC',
               mode='lines', line=dict(color='#271f30', dash='dot'))
    nncen=go.Scatter(x=time, y=energy_nnc, name='NODEC',
               mode='lines', line=dict(color='#ff9f00', dash='dot'))
    fig_energies = go.Figure([ocen,nncen])
    figs = make_subplots(1, 2)
    for trace in fig_trajectories.data:
        figs.append_trace(trace, 1, 1)
    for trace in fig_energies.data:
        trace.showlegend=False
        trace.xaxis = 'x2'
        trace.yaxis = 'y2'
        figs.append_trace(trace, 1, 2)
    figs.update_layout(base_temp)
    figs.update_xaxes(axis_temp)
    figs.update_yaxes(axis_temp)
    figs.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    figs.layout.height=400
    return figs, fig_trajectories, fig_energies


def animate_energy(alldf, linear_dynamics, oc_baseline, x0, T, n_points=40):
    trajectory = evaluate_trajectory(linear_dynamics,
                                     oc_baseline,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / 40)
                                     )
    oc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    alldf['animation_frame'] = alldf['sample_id'].astype(str) + ' ' + alldf['lr'].astype(str)
    mapper = dict((v, k) for (k, v) in enumerate(alldf['animation_frame'].unique().tolist()))
    alldf['animation_frame'] = alldf['animation_frame'].map(mapper)
    alldf['energy'] = (alldf['u'] ** 2) * T / 39
    alldf['energy'] = alldf.groupby(['lr', 'sample_id'])[['lr', 'sample_id', 'energy']].cumsum()
    alldf['exact_time'] = alldf['time'] * T / 39
    oc_trajectory['energy'] = (oc_trajectory['u'] ** 2) * T / 39
    fig_energy_anim = px.line(alldf, x='exact_time', y='energy',
                   animation_frame='animation_frame',

                   animation_group='lr',
                   color='lr',

                   color_discrete_map={
                       0.1: 'red',
                       0.01: 'green',
                       0.001: 'blue'
                   }
                   )
    fig_oc = go.Scatter(x=oc_trajectory.index * (T / 39), y=oc_trajectory['energy'].cumsum(),
                        line=dict(color='#271f30', dash='dot'), name='OC')
    fig_energy_anim.add_trace(fig_oc)
    return fig_energy_anim


def animate_trajectories(alldf,linear_dynamics, oc_baseline, nnc,  x0, x_target, T):
    n_points=int(alldf['time'].max()+1)
    trajectory = evaluate_trajectory(linear_dynamics,
                                     oc_baseline,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    oc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    trajectory = evaluate_trajectory(linear_dynamics,
                                     nnc,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    nnc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    x1_min, x1_max = alldf['x1'].min(), alldf['x1'].max()
    x2_min, x2_max = alldf['x2'].min(), alldf['x2'].max()
    fig_trajectories = plot_trajectory_comparison(linear_dynamics,
                               x0,
                               x_target,
                               nnc_trajectory,
                               oc_trajectory,
                               x1_min,
                               x1_max,
                               x2_min,
                               x2_max
                               )



    fig = px.line(alldf, x='x1', y='x2',
                  animation_frame='animation_frame',

                  animation_group='lr',
                  color='lr',

                  color_discrete_map={
                      0.1: 'red',
                      0.01: 'green',
                  }
                  )

    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode="lines",
                             line=dict(color='green'),
                             name='lr=0.01'
                             ))

    fig.add_trace(fig_trajectories.data[0])
    fig.add_trace(fig_trajectories.data[1])
    fig.add_trace(fig_trajectories.data[3])
    fig.add_trace(fig_trajectories.data[2])
    return fig

def grand_animation(alldf,linear_dynamics, oc_baseline, nnc,  x0, x_target, T, frame_duration=50):
    x1_min, x1_max = alldf['x1'].min(), alldf['x1'].max()
    x2_min, x2_max = alldf['x2'].min(), alldf['x2'].max()
    # fig
    fig_energy_anim = animate_energy(alldf, linear_dynamics, oc_baseline, x0, T, n_points=40)
    fig_traj_anim = animate_trajectories(alldf, linear_dynamics, oc_baseline, nnc, x0,
                                         x_target, T)
    fig_gran_anim = make_subplots(1, 2)
    for trace in fig_traj_anim.data:
        fig_gran_anim.append_trace(trace, 1, 1)
    for trace in fig_energy_anim.data[1:]:
        trace.xaxis = 'x2'
        trace.yaxis = 'y2'
        trace.showlegend = False
        fig_gran_anim.append_trace(trace, 1, 2)
    fig_gran_anim.layout.xaxis.range = [x1_min, x2_max]
    fig_gran_anim.layout.yaxis.range = [x2_min - 0.1, x2_max]
    fig_gran_anim = fig_gran_anim.update_layout(plot_base_template(
        x1_min - 0.1 * abs(x1_min),
        x1_max,
        x2_min - 0.1 * abs(x2_min),
        x2_max
    ))
    fig_gran_anim.layout.sliders = fig_traj_anim.layout.sliders
    fig_gran_anim.layout.updatemenus = fig_traj_anim.layout.updatemenus
    fig_gran_anim.layout.updatemenus[0]['buttons'][0].args[1]['transition']['easing'] = 'linear'
    fig_gran_anim.layout.updatemenus[0]['buttons'][0].args[1]['transition']['duration'] = 1
    fig_gran_anim.frames = fig_traj_anim.frames
    for i, frame in enumerate(fig_energy_anim.frames):
        frame.data[0].xaxis = 'x2'
        frame.data[0].yaxis = 'y2'
        fig_gran_anim.frames[i].data = list(fig_traj_anim.frames[i].data) + list(frame.data)

    fig_gran_anim.layout.width = 800
    fig_gran_anim.layout.height = 400
    fig_gran_anim.update_layout(base_temp)
    fig_gran_anim.update_xaxes(axis_temp)
    fig_gran_anim.update_yaxes(axis_temp)
    fig_gran_anim.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig_gran_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
    return fig_gran_anim
