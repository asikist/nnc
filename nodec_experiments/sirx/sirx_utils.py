from torchdiffeq import odeint
import torch
import numpy as np
import pandas as pd
import plotly.express as px

from plotly import graph_objects as go

from plotly.subplots import make_subplots


def logmap(plotdata, cmin, cmax, caxis_name, tickvals, ticktext, colorscale='Viridis',
           show_scale=True):
    """
    Special heatmap plot for sirx states over lattice
    :param plotdata: the data to plot
    :param cmin: the min color value
    :param cmax: the max color value
    :param caxis_name: the colorscale axis name
    :param tickvals: the tick value stops on the color axis
    :param ticktext: the tick text on tick stops ion the color axis
    :param colorscale: the color scale to use, a string value based on plotly colorscales
    :param show_scale: whether to show the scale or not
    :return: the plotly figure,
    """
    proposed_layout = dict(font=dict(family='Times New Roman',
                                     size=18),
                           xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           width=200,
                           height=200,
                           margin=dict(t=0, l=0, r=70, b=0),
                           # coloraxis = dict(showscale=False)
                           )

    fig = px.imshow(plotdata, color_continuous_scale=colorscale, zmin=cmin, zmax=cmax)
    fig.update_layout(proposed_layout)

    fig.update_layout(
        plot_bgcolor='white',
        coloraxis_colorbar=dict(
            title=caxis_name,
            titleside='right',
            thicknessmode="pixels",
            thickness=7,
            lenmode="pixels",
            len=110,
            tickvals=tickvals,
            ticktext=ticktext,
            ticks="outside",
        ))
    return fig


def trajectory_eval(sir, x_0, model=None, dt=0.001, T=5, method='adams'):
    """
    A simple trajectory evaluator for SIRX similar to:
    the ``` nnc.helpers.torch_utils.evaluators.FixedInteractionEvaluator```
    :param sir: the sirx dynamics
    :param x_0: the initial state
    :param model: the control model
    :param dt: the time interval between interactions
    :param T: the total time
    :param method: the ode solver method, based on torchdiffeq.odeint documentation.
    :return: state and control trajectories over all nodes
    """
    all_u = []
    all_x = [x_0]
    x = x_0
    t = 0
    for i in range(int(T // dt)):
        if model is not None:
            u = model(x=x, t=torch.tensor([t, t + dt], device=x_0.device)[0])
            u = u.detach()
        else:
            u = torch.zeros_like(x[:, sir.n_nodes])
        # sanity/insanity checks
        if method != 'rk4':
            options = {}
            if method == 'adams':
                # default constructor value is also True, but we need to make sure even if it changes.
                options['implicit'] = True
            x = odeint(lambda t, x: sir(t, x, u=u), x, torch.tensor([t, t + dt]),
                       method='implicit_adams')[-1]
        else:
            x = odeint(lambda t, x: sir(t, x, u=u), x, torch.tensor([t, t + dt]), method='rk4')[-1]
        t += dt
        all_u.append(u)
        all_x.append(x)
    return torch.cat(all_x).cpu().numpy(), torch.cat(all_u).cpu().numpy()


def sirx_curves(all_xnp, selection=None):
    """
    Plots of all SIRX curves
    :param all_xnp: numpy array with all SIRX states.
    :param selection: The list with node indices to plot from, a subgraph selection.
    :return:
    """
    n_nodes = int(all_xnp.shape[-1] / 4)
    if selection is None:
        selection = np.arange(0, n_nodes)
    allinfected = all_xnp[:, :n_nodes][:, selection].mean(-1)
    allrecovered = all_xnp[:, 2 * n_nodes:3 * n_nodes][:, selection].mean(-1)
    allsusc = all_xnp[:, n_nodes:2 * n_nodes][:, selection].mean(-1)
    allcont = all_xnp[:, 3 * n_nodes:4 * n_nodes][:, selection].mean(-1)

    line_colors = [
        '#e66101',
        '#fdb863',
        '#b2abd2',
        '#5e3c99'
    ]

    a = px.line(y=allinfected, x=np.linspace(0, 5, allinfected.shape[0]), render_mode='svg')
    a.data[0].line.color = line_colors[0]
    a.data[0].name = 'infected'
    a.data[0].showlegend = True

    b = px.line(y=allrecovered, x=np.linspace(0, 5, allinfected.shape[0]), render_mode='svg')
    b.data[0].line.color = line_colors[2]
    b.data[0].name = 'recovered'
    b.data[0].showlegend = True

    c = px.line(y=allsusc, x=np.linspace(0, 5, allinfected.shape[0]), render_mode='svg')
    c.data[0].line.color = line_colors[1]
    c.data[0].name = 'susceptible'
    c.data[0].showlegend = True

    d = px.line(y=allcont, x=np.linspace(0, 5, allinfected.shape[0]), render_mode='svg')
    d.data[0].line.color = line_colors[3]
    d.data[0].name = 'contained'
    d.data[0].showlegend = True

    fig = go.Figure([a.data[0], b.data[0], c.data[0], d.data[0]])
    fig.update_layout(template='simple_white',
                      paper_bgcolor='rgba(255,255,255,1)',
                      plot_bgcolor='rgba(255,255,255,1)',
                      xaxis=dict(showgrid=False, title='Time'),
                      yaxis=dict(showgrid=False, title='Mean Value',
                                 tickmode='array',
                                 tickvals=[0, 0.25, 0.5, 0.75, 1],
                                 dtick=0.2,
                                 nticks=5,
                                 # plotoos = None
                                 # type='log', tickvals=[10e-6,10e-4, 10e-2, 10e-2, 1]
                                 ),
                      width=250,
                      height=250,
                      # margin
                      # title='Phase Plot: Optimal MSE Control',
                      font=dict(family='Times New Roman', size=12),
                      margin=dict(t=0, l=0, r=0, b=0),
                      legend=dict(
                          x=-0.27,
                          y=1.5,
                          orientation='h',
                          traceorder="normal",
                          font=dict(
                              family="Times New Roman",
                              color="black"
                          ),
                          bgcolor=None,
                          bordercolor='rgba(0,0,0,0)',
                          borderwidth=1
                      )
                      )
    return fig


def comparison(name, x, data: dict, colors: dict):
    """
    Comparison over state averages for different models
    :param name: The name of the plot
    :param x: xaxis index values, often it is the time, has to be same length as the data value
    lists
    :param data: The data dictionary with keys the controller names and values the state outcomes.
    :param colors: color dictionary with key that matches the data key and value a color code.
    :return:
    """
    df = pd.DataFrame(data)
    traces = []
    for column in df.columns.values:
        y = df[column].values
        dash = 'solid'
        if column == 'RL':
            dash = 'dot'
        traces.append(go.Scatter(x=x, y=y, name=column, marker=dict(color=colors[column]),
                                 line=dict(dash=dash)))
    fig = go.Figure(traces)
    fig.update_layout(template='simple_white',
                      paper_bgcolor='rgba(255,255,255,1)',
                      plot_bgcolor='rgba(255,255,255,1)',
                      xaxis=dict(showgrid=False, title='Time'),
                      yaxis=dict(showgrid=False, title=name,
                                 tickmode='array',
                                 # tickvals=[0, 0.25, 0.5, 0.75, 1],
                                 # dtick = 0.2,
                                 # nticks=5,
                                 # plotoos = None
                                 # type='log',
                                 # exponentformat='power'
                                 # tickvals=[10e-6,10e-4, 10e-2, 10e-2, 1]
                                 ),
                      width=200,
                      height=200,
                      # margin
                      # title='Phase Plot: Optimal MSE Control',
                      font=dict(family='Times New Roman', size=14),
                      margin=dict(t=0, l=0, r=0, b=0),
                      legend=dict(
                          x=-0.27,
                          y=1.5,
                          orientation='h',
                          traceorder="normal",
                          font=dict(
                              family="Times New Roman",
                              color="black"
                          ),
                          bgcolor=None,
                          bordercolor='rgba(0,0,0,0)',
                          borderwidth=1
                      )
                      )
    return fig


def stack_plot_grid(name, plot_grid, colorscale=px.colors.sequential.Plasma, shared_yaxes=True,
                    shared_xaxes=True, horizontal_spacing=0.015 / 2, vertical_spacing=0.015 / 2):
    """
    A plot grid that combines many figures, Ideal for heatmap stacks
    :param name: The name of the plot
    :param plot_grid: a list of rows and columns values to define the grid
    :param colorscale: the colorscale to use from plotly.colors
    :param shared_yaxes: whether xaxes are shared across plots
    :param shared_xaxes: whether yaxes are shared across plots
    :param horizontal_spacing: spacing between plots horizontally
    :param vertical_spacing: spcacing between plots vertically
    :return: The figure containing the plots.
    """
    plot_grid = np.array(plot_grid)
    columns = plot_grid.shape[0]
    if len(plot_grid.shape) == 2:
        rows = plot_grid.shape[0]
        columns = plot_grid.shape[1]
    else:
        rows = 1
    figs = make_subplots(rows=rows, cols=columns,
                         shared_yaxes=shared_yaxes,
                         shared_xaxes=shared_xaxes,
                         horizontal_spacing=horizontal_spacing,
                         vertical_spacing=vertical_spacing,
                         )
    for col_i in range(columns):
        for row_i in range(rows):
            if len(plot_grid.shape) < 2:
                fig = plot_grid[col_i]
            else:
                fig = plot_grid[row_i, col_i]
            figs.append_trace(fig.data[0], row_i + 1, col_i + 1)

    figs.update_layout(
        font=dict(family='Times New Roman',
                  size=20),
        xaxis=dict(visible=False),
        plot_bgcolor='white',
        coloraxis_colorbar=dict(
            # title=dict(text='$x(t)$'),
            # titleside='top',
            thicknessmode="pixels",
            thickness=24,
            tickmode='array',
            len=0.8,
            x=1.01,
            nticks=7,
            ticks="outside",
        ),
        coloraxis=dict(cmax=1),
        margin=dict(r=100, l=0, t=0, b=0),
    )
    for i in range((columns * rows) + 1):
        figs.layout['xaxis' + [str(i), ''][i == 0]].visible = False
        figs.layout['yaxis' + [str(i), ''][i == 0]].visible = False
    # figs.write_image(outfolder+'inf.pdf')
    figs.layout.annotations = [dict(text=name, x=1.09, y=0.95, textangle=0)]
    figs.update_annotations(dict(
        xref="paper",
        yref="paper",
        showarrow=False,
        ax=0,
        ay=0,
        font=dict(
            color="black",
            size=20
        )
    ))
    figs.update_layout(coloraxis_colorscale=colorscale)

    return figs


def heats_for_steps(name, z_data, colorscale, timesteps, zmin=0, zmax=1,
                    ztickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], zticktext=None):
    """
    The grtid containing many heatmaps based on a list of color data
    :param name: The name of the plot
    :param z_data: the list containing data to be plotted as heatmaps
    :param colorscale: the colorscale to use from plotly.colors
    :param timesteps: the timesteps across which we samplethe plors, zdata 0 dimension
    :param zmin: the minimuim z axis value to normalize color
    :param zmax: the maxium z axis value to normalize color
    :param ztickvals: the tick values of the colorscale stops
    :param zticktext: the tick values displayed on colorscale stops as labels
    :return:
    """
    res = []
    j = 0
    for step in timesteps:
        side_size = int(np.sqrt(z_data.shape[-1]))
        zdata = z_data[step].reshape(32, 32).copy()

        fig = logmap(zdata, zmin, zmax, name, tickvals=ztickvals, ticktext=zticktext,
                     colorscale=colorscale)
        if j == len(timesteps) - 1:
            fig.update_layout(coloraxis=dict(showscale=True), margin=dict(r=120), width=320)
        else:
            fig.update_layout(coloraxis=dict(showscale=False), margin=dict(r=0), width=200)
        j += 1
        res.append(fig)
    return res
