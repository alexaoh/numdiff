"""Some utilities for plotting solutions of the heat equation in 1d."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np

def three_dim_plot(*, xv, tv, I, label):
    """Plot solution (I) in a 3 dimensions."""
    fig = plt.figure()
    fig.suptitle(label)
    ax = fig.gca(projection="3d")
    ax.plot_surface(xv, tv, I, cmap="seismic")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("Intensiity")
    plt.show()

def sub(*, x, I, t, L, t_index, label):
    """Make subplots at given timestep indices."""
    subplts = len(t_index)
    fig, axs = plt.subplots(int(subplts/2), int(subplts/2))
    precision = 3 # Precision to round off the times in the titles below. 
    axs[0, 0].plot(x, I[:, t_index[0]], 'o', markerfacecolor='none')
    axs[0, 0].set_title(f'$t$ = {round(t[t_index[0]], precision)}')
    axs[0, 1].plot(x, I[:, t_index[1]], 'o', markerfacecolor='none')
    axs[0, 1].set_title(f'$t$ = {round(t[t_index[1]], precision)}')
    axs[1, 0].plot(x, I[:, t_index[2]], 'o', markerfacecolor='none')
    axs[1, 0].set_title(f'$t$ = {round(t[t_index[2]], precision)}')
    axs[1, 0].set_xlabel('$x$') 
    axs[1, 0].set_ylabel('Intensity')
    axs[1, 1].plot(x, I[:, t_index[3]], 'o', markerfacecolor='none')
    axs[1, 1].set_title(f'$t$ = {round(t[t_index[3]], precision)}')

    plt.setp(axs,  xlim=(0,L), ylim=(0,np.amax(I)+1))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(label)
    plt.show()
    