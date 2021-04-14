"""Utility functions that is imported when in need of plotting solutions."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d_sol(U, xv, yv, Uan = False, savename = False):
    """Plot numerical solution (and optionally analytical) for task 3: 2D Laplace. """
    # Taken from three_dim_plot and heavily modified later (could be added to utilities or something in the end).
    fig = plt.figure()
    if callable(Uan):
        fig.suptitle("Num Sol, M = "+str(U.shape[0]-2)+", + An Sol")
    else: 
        fig.suptitle("Num Sol, Mx = "+str(U.shape[1]-2)+" My = "+str(U.shape[0]-2))
    ax = fig.gca(projection="3d")
    ax.view_init(azim=55, elev=15) # Added some rotation to the figure. 
    surface = ax.plot_surface(xv, yv, U, cmap="seismic") 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("Intensity")
    #fig.colorbar(surface, shrink=0.5, aspect=5)
    if callable(Uan):
        x = y = np.linspace(0, 1, 1000) # Gives three internal points + boundaries.
        xv, yv = np.meshgrid(x, y)
        surface2 = ax.plot_surface(xv, yv, Uan(xv, yv), cmap="Greys", alpha = 0.7)
        #fig.colorbar(surface2, shrink=0.5, aspect=5)
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()
