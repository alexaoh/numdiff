"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def num_solution_M3(save = False):
    """Numerical solution of the 2D Laplace with five point stencil, with M_x = M_y = 9 =: M.
    
    This solution matches the example from the report. 
    """

    M = 3

    # Construct A. 
    data = np.array([np.full(M**2, 1), np.full(M**2, -4), np.full(M**2, 1), 
                        np.full(M**2, 1), np.full(M**2, 1)])
    diags = np.array([-1, 0, 1, -3, 3])
    A = spdiags(data, diags, M**2, M**2).toarray()

    # Change some elements to match the correct linear system. 
    A[3, 2], A[2, 3], A[6, 5], A[5, 6] = 0, 0, 0, 0

    # Construct F.
    F = np.zeros(M**2)
    F[2] = -np.sin(2*np.pi*1/4)
    F[5] = -np.sin(2*np.pi*2/4)
    F[8] = -np.sin(2*np.pi*3/4)

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)

    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    U[-1, :] = np.array([0, np.sin(2*np.pi*1/4), np.sin(2*np.pi*2/4), np.sin(2*np.pi*3/4), 0]) 
    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/3)):
        U[1, i+1] = Usol[i]
        U[2, i+1] = Usol[i+3]
        U[3, i+1] = Usol[i+6]

    # Taken from three_dim_plot (could be added to utilities or something in the end).
    fig = plt.figure()
    fig.suptitle("Num Sol, M = 3")
    ax = fig.gca(projection="3d")
    ax.view_init(azim=-45, elev=25) # Added some rotation to the figure. 
    ax.plot_surface(xv, yv, U, cmap="seismic")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("Intensity")
    if save:
        plt.savefig("NumSolM3Task3.pdf")
    plt.show()

num_solution_M3()
