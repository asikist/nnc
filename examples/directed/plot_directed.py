import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
                               
# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True,
    'text.latex.preamble': [r"\usepackage{bm}", r"\usepackage{mathtools}"]}
    # tell matplotlib about your params

rcParams.update(params)
# set nice figure sizes
fig_width_pt = 510    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [fig_width, 0.5*fig_height]
rcParams.update({'figure.figsize': fig_size})

data_energy = np.loadtxt("energies_directed5.csv")
oc_trajectories = np.loadtxt("oc_trajectory_directed5.csv")
nnc_trajectories = np.loadtxt("nnc_trajectory_directed5.csv")

colors = ['#003f5c', '#7a5195' , '#ef5675', '#ffa600']

fig, ax = plt.subplots(ncols = 2)

ax[0].text(0.02*1, 0.89*20-10, r'(a)', bbox=dict(facecolor='white', alpha=0.7, edgecolor = 'None', boxstyle="square,pad=0."))

ax[0].plot(data_energy[:,0],oc_trajectories[:,0], color = 'k', ls = '--', alpha = 0.6, zorder=100)
ax[0].plot(data_energy[:,0],oc_trajectories[:,1], color = 'k', ls = '--', alpha = 0.6, zorder=100)
ax[0].plot(data_energy[:,0],oc_trajectories[:,2], color = 'k', ls = '--', alpha = 0.6, zorder=100)
ax[0].plot(data_energy[:,0],oc_trajectories[:,3], color = 'k', ls = '--', alpha = 0.6, zorder=100)

ax[0].plot(data_energy[:,0],nnc_trajectories[:,0], label=r'$x_1(t)$')
ax[0].plot(data_energy[:,0],nnc_trajectories[:,1], label=r'$x_2(t)$')
ax[0].plot(data_energy[:,0],nnc_trajectories[:,2], label=r'$x_3(t)$')
ax[0].plot(data_energy[:,0],nnc_trajectories[:,3], label=r'$x_3(t)$')

ax[0].set_xlim(0,1)
#ax[0].set_ylim(-10,10)
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x_i(t)$')
ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[0].legend(loc=1,frameon=False,fontsize=8,ncol=3)

ax[1].text(0.02*1, 0.89*800, r'(b)', bbox=dict(facecolor='white', alpha=0.7, edgecolor = 'None', boxstyle="square,pad=0."))

ax[1].plot(data_energy[:,0],data_energy[:,1], color = colors[-1])
ax[1].plot(data_energy[:,0],data_energy[:,2], color = 'k', ls = '--', alpha = 0.6)
ax[1].hlines(data_energy[:,2][-1], 0, 1, linestyle = '--', colors = '#e84a5f', zorder = 100)
ax[1].text(0.13, 680, r'OC control energy', color = '#e84a5f', fontsize = 8)
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$E_t[ {\bm u} ]$')
ax[1].set_ylim(0,800)
ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
#ax[1].set_yticks([0,0.4,0.8,1.2,1.6])

plt.tight_layout()
plt.margins(0,0)
#plt.savefig('three_state_system_directed_absorbing.svg', dpi = 480, pad_inches = 0.05)

plt.show()
