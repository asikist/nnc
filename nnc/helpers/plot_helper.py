from copy import deepcopy

import plotly
import plotly.graph_objs as go
import networkx as nx
import plotly.express as px
from plotly.subplots import make_subplots
import colour
import re
from plotly.colors import hex_to_rgb
from plotly import graph_objects as go
import torch
import scipy
import pandas as  pd
from math import floor, log10

"""
Basic plot utilities used for the experiments.
Custom plot utilities are also found in corresponding experiment folders.
"""

base_layout = dict(template='plotly_white',
                   font=dict(family='Times New Roman', size=10),
                   paper_bgcolor='rgba(255,255,255,1)',
                   plot_bgcolor='rgba(255,255,255,1)',
                   xaxis=dict(showgrid=False, zeroline=False, showline=True,
                              linewidth=1, linecolor='black', mirror=True),
                   yaxis=dict(showgrid=False, zeroline=False, showline=True,
                              linewidth=1, linecolor='black', mirror=True),
                   )


def round_to_text(value, rounding_precision):
    """
    A simple method that formats values to nicestrings
    :param value: the value to format
    :param rounding_precision: the required rounding precision
    :return:
    """
    return '{:.2f}'.format(round(value, rounding_precision))


class ColorRegistry:
    """
    A custom color registry to make clean plots for SIRX.
    The colors are choosen based on:
    `https://coolors.co/`
    """
    nodec = '#5e3c99'#'#61E8E1'
    oc = '#e66101'#'#773344'
    rl = '#4392F1'
    random = '#F092DD'
    constant = '#99C1B9'


def getIfromRGB(rgb):
    """
    Converts rgb tuple to integer
    :param rgb: the rgb tuple n 255 scale
    :return: the integer
    """
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint


def to_rgba(color, alpha = 0.5):
    """
    Adds opacity to an rgb string
    :param color: The color string
    :param alpha: the require opacity level in [0,1]
    :return: The color string in rgba(r,g,b,a) format
    """
    o = list(colour.Color(color).get_rgb())
    o.append(alpha)
    return 'rgba'+str(tuple(o))


def generate_annotation(x, y, text, color='black', fsize=12, textangle=0, xref='paper',
                       yref='paper', align='left'):
    """
    Generates a plotly annotation in the form of dictionary.
    It can be directly be appended to the layout.annotations.
    :param x: x coordinate in plot
    :param y: y coordinate in plot
    :param text: text content
    :param color: color of text
    :param fsize: font size
    :param textangle: angle of text
    :param xref: reference to 'paper'/canvas or 'plot'/axis coordinates, more details in `plotly` documentation
    :param yref: reference to 'paper'/canvas or 'plot'/axis coordinate,  more details in `plotly` documentation
    :return: The annotation object to append in the `layout` of the figure.
    """
    res = dict(
        y=y,
        x=x,
        showarrow=False,
        textangle=textangle,
        text=text,
        font=dict(
            color=color,
            size=fsize
        ),
        align = align,
        xref=xref,
        yref=yref
    )
    return res


def add_line(x1, y1, x2=None, y2=None, color='white', width=2, dash='solid', t='line', xref='paper',
             yref='paper'):
    """
    Add a line shape. Please check `plotly.graph_objects.Layout.add_shape` documentation for more details on the inputs.
    :param x1: starting `x` coordinate
    :param y1: starting `y` coordinate
    :param x2: last `x` coordinate
    :param y2:  last `y` coordinate
    :param color: the color of the line
    :param width: the width of the line
    :param dash: whether the line is 'solid', 'dashed' etc.
    :param t: the type of the shape, if you change it from line it can become another shaper.
    :param xref:  reference to 'paper'/canvas or 'plot'/axis coordinates, more details in `plotly` documentation
    :param yref: reference to 'paper'/canvas or 'plot'/axis coordinates, more details in `plotly` documentation
    :return:
    """
    res = {
        'type': t,
        'y0': y1,
        'x0': x1,
        'y1': y2 or y1,
        'x1': x2 or x1,
        'line': {
            'color': color,
            'width': width,
            'dash': dash,
        },
        'xref': xref,
        'yref': yref
    }
    return res


def add_shape(x1, y1, x2=None, y2=None, shape_type='line', color='black', width=1, dash='solid', \
                                                                                    xref='paper',
             yref='paper'):
    """
    Add a shape. Please check `plotly.graph_objects.Layout.add_shape` documentation for more details on the inputs.
    :param x1: starting `x` coordinate
    :param y1: starting `y` coordinate
    :param x2: last `x` coordinate
    :param y2:  last `y` coordinate
    :param color: the color of the line
    :param width: the width of the line
    :param dash: whether the line is 'solid', 'dashed' etc.
    :param t: the type of the shape, if you change it from line it can become another shaper.
    :param xref:  reference to 'paper'/canvas or 'plot'/axis coordinates, more details in `plotly` documentation
    :param yref: reference to 'paper'/canvas or 'plot'/axis coordinates, more details in `plotly` documentation
    :return:
    """
    res = {
        'type': shape_type,
        'y0': y1,
        'x0': x1,
        'y1': y2 or y1,
        'x1': x2 or x1,
        'line' : {
            'color': color,
            'width': width,
            'dash': dash,
        },
        'xref': xref,
        'yref': yref
    }
    return res


def combines_figs(rows, columns, figs):
    """
    Combine multiple figures in a row to column format.
    :param rows: The number of grid rows
    :param columns: The numebr of grid columns
    :param figs: The list of figures allocated though a row sequentially, and once a row is
    filled, the iterator proceeds to the next (row-major-order).
    :return:
    """
    fig = make_subplots(rows=rows, cols=columns)
    for i, f in enumerate(figs):
        col = (i)%columns + 1
        row = int((i-(i)%columns)//rows) + 1
        for data in f.data:
            fig.add_trace(data,
                 row=row, col=col)
    return fig


def square_lattice_heatmap(reached_state,
                           color_scale = plotly.colors.sequential.Plasma,
                           zmin = -1,
                           zmax = 1,
                           rmin = None,
                           rmax= None,
                           rounding_prec = 2,
                           low_outlier_color = '#000000',
                           high_outlier_color = '#ffffff',
                           tickvals = None,
                           ticktext = None,
                          ):
    """
    A heatmap generator for states on square lattices.
    :param reached_state: The state to plot
    :param color_scale: the color scale to use
    :param zmin: The minimum non-outlier color value. Values lower than this will be assigned an
    outlier color.
    :param zmax: The max non-outlier color value.  Values higher than this will be assigned an
    outlier color.
    :param rmin: The min value in the matrix. If none it will be calculated automatically,
    otherwise it is used as reference (label construction). Does not need to be in the data.
    :param rmax: The max value in the matrix. If none it will be calculated automatically,
    otherwise it is used as reference (label construction). Does not need to be in the data.
    :param rounding_prec: The rounding precision of the values.
    :param low_outlier_color: The color for low outliers.
    :param high_outlier_color: The color for high outliers.
    :param tickvals: The tick values in the colorscale (points to place the ticks on)
    :param ticktext: The text values or labels on the tick points.
    :return: The plotly heatmap figure.
    """

    if rmin is None:
        rmin = reached_state.min().item()
    if rmax is None:
        rmax = reached_state.max().item()
    if tickvals is None:
        tickvals = [zmin, (zmin+zmax)/2, zmax]
    if ticktext is None:
        ticktext = [round_to_text(zmin, rounding_prec), round_to_text((zmin+zmax)/2, rounding_prec),
                    round_to_text(zmax, rounding_prec)]
    original_scale = deepcopy(color_scale)
    is_hex = original_scale[0].startswith('#')
    if is_hex:
        original_scale = ['rgb' + str(hex_to_rgb(c)) for c in original_scale]
    extra_steps = 0
    if rmin < zmin:
        if low_outlier_color.startswith('#'):
            low_outlier_color = 'rgb' + str(hex_to_rgb(low_outlier_color))
        original_scale.insert(0, low_outlier_color)
        extra_steps+=1
    if rmax > zmax:
        if high_outlier_color.startswith('#'):
            high_outlier_color = 'rgb' + str(hex_to_rgb(high_outlier_color))
        original_scale.append(high_outlier_color)
        extra_steps+=1


    n_csteps = len(original_scale)


    first_color = original_scale[0]
    last_color = original_scale[-1]

    toplot = reached_state.clone()
    if rmin < zmin:
        scale_min = zmin - (1.0/(n_csteps-1-extra_steps))*(zmax-zmin)
        toplot[toplot < zmin] = scale_min
        tickvals.insert(0, scale_min)
        ticktext.insert(0 , round_to_text(rmin, rounding_prec))
    else:
        scale_min = zmin
    if rmax > zmax:
        scale_max = zmax + (1.0/(n_csteps-1-extra_steps))*(zmax-zmin)
        toplot[toplot > zmax] = scale_max
        tickvals.append(scale_max)
        ticktext.append(round_to_text(rmax, rounding_prec))
    else:
        scale_max = zmax

    all_steps = torch.linspace(0, 1, n_csteps).cpu().numpy().tolist()
    resscale = list(list(a) for a in zip(all_steps, original_scale))
    if rmin < zmin:
        resscale.insert(1, [all_steps[1], low_outlier_color])
    if rmax > zmax:
        resscale.insert(-1, [all_steps[-2], high_outlier_color])

    colorbar=dict(title=None,#r"$\quad \quad \, x$ <div>a<\div>",
                  titleside="bottom",
                  tickvals= tickvals ,
                  ticktext= ticktext,
                  ticks="inside",
                  len=1.07,
                  outlinecolor='black',
                  outlinewidth=1,
                  thickness= 5,
                  xpad=2
                 )
    trace = go.Heatmap(z=toplot.reshape([32,32]),  colorbar = colorbar, colorscale=resscale, zmin=scale_min, zmax=scale_max)
    fig = go.Figure(trace)
    #fig.layout.xaxis.visible = False
    #fig.layout.yaxis.visible = False
    fig.layout.xaxis.nticks = 0
    fig.layout.yaxis.nticks = 0
    fig.layout.xaxis.tickvals = []
    fig.layout.yaxis.tickvals = []
    fig.layout.font.family = 'Times New Roman'
    fig.layout.font.size = 10
    fig.layout.width = 120#210
    fig.layout.height = 90#150
    fig.layout.margin = dict(t=1, l=1, r=35, b=1)
    fig = fig.update_xaxes(showline=True,
                             linewidth=1, linecolor='black', mirror=True)
    fig = fig.update_yaxes(showline=True,
                             linewidth=1, linecolor='black', mirror=True)
    return fig


def trendplot(x1, x2, ax1, ax2,
              render_mode='svg',
              marker_color='rgba(178,171,210, 0.5)',
              line_color='rgba(230,97,1,0.8)',
              show_stats=True
              ):
    """
    A scatter plot between a pair of 1-D point clouds equipped with a trendline based on OLS.
    :param x1: The collection of the first point cloud.
    :param x2: The collection of the second point cloud.
    :param ax1: Name of axis corresponding to `x1`, i.e. horizontal axis.
    :param ax2:  Name of axis corresponding to `x2`, i.e. vertical axis.
    :param render_mode: whether to render via 'svg' or 'webgl' engines.
    :param marker_color: The color of the points.
    :param line_color: The OLS line color
    :param show_stats: whether to show line slope and spearman correlation.
    :return: The plotly figure.
    """
    fig = px.scatter(x=x1, y=x2, trendline='ols', trendline_color_override='red',
                     render_mode=render_mode,
                     # marginal_x = 'violin',
                     # marginal_y='violin'
                     )
    corr, pval = scipy.stats.spearmanr(x1, x2)
    results = px.get_trendline_results(fig)
    regr_table = (results.iloc[0][0].summary().tables[1])
    regr_df = pd.DataFrame(regr_table.data[1:], columns=regr_table.data[0]).set_index('')
    slope = float(regr_df.loc['x1', 'coef'])

    fig.data[0].marker.size = 2
    fig.data[0].mode = 'markers'

    fig.data[0].marker.color = marker_color
    fig.data[1].marker.color = line_color
    if show_stats:
        fig.update_layout(annotations=[
            dict(
                x=-0.3,
                y=1.4,
                xref="paper",
                showarrow=False,
                ax=0,
                align="right",
                ay=0,
                yref="paper",
                text='slope: ' + '{:.2f}'.format(round(slope, 2)) +
                     ",<br>corr: " + '{:.2f}'.format(round(corr, 2))
                     #+ ", p: " +'{:.2f}'.format(round(pval, 2)),
            )
        ])
    fig.update_layout(base_layout)
    fig.layout.xaxis.title = ax1
    fig.layout.yaxis.title = ax2
    fig.update_layout(coloraxis=dict(showscale=False),
                      width=140,
                      height=162,
                      margin=dict(t=40,
                                  l=20,
                                  r=1,
                                  b=0
                                  )
                      )
    return fig


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    :param num: the number to format
    :param decimal_digits: the number of decimal digits to use
    :param precision: the precision of the exponent
    :param exponent: The exponent, which can also be calculated automatically.
    :return:

    * From https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten
     -formatting
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    if coeff != 1:
        return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
    else:
        return r"$10^{{{:d}}}$".format(exponent)
