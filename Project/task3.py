"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def num_solution_M3():
    """Numerical solution of the 2D Laplace with five point stencil, with M_x = M_y = 3 =: M.
    
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


    return U, xv, yv

def num_solution_M9():
    """Numerical solution of the 2D Laplace with five point stencil, with M_x = M_y = 9 =: M.
    
    This method has 9*9 = 81 unknown internal grid points.  
    """

    M = 9

    # Construct A. 
    data = np.array([np.full(M**2, 1), np.full(M**2, -4), np.full(M**2, 1), 
                        np.full(M**2, 1), np.full(M**2, 1)])
    diags = np.array([-1, 0, 1, -9, 9])
    A = spdiags(data, diags, M**2, M**2).toarray()

    # Change some elements to match the correct linear system. 
    A[9, 8], A[8, 9], A[18, 17], A[17, 18] = 0, 0, 0, 0
    A[27, 26], A[26, 27], A[36, 35], A[35, 36] = 0, 0, 0, 0
    A[45, 44], A[44, 45], A[54, 53], A[53, 54] = 0, 0, 0, 0
    A[63, 62], A[62, 63], A[72, 71], A[71, 72] = 0, 0, 0, 0

    # Construct F.
    F = np.zeros(M**2)
    F[8] = -np.sin(2*np.pi*1/10)
    F[17] = -np.sin(2*np.pi*2/10)
    F[26] = -np.sin(2*np.pi*3/10)
    F[35] = -np.sin(2*np.pi*4/10)
    F[44] = -np.sin(2*np.pi*5/10)
    F[53] = -np.sin(2*np.pi*6/10)
    F[62] = -np.sin(2*np.pi*7/10)
    F[71] = -np.sin(2*np.pi*8/10)
    F[80] = -np.sin(2*np.pi*9/10)

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)
    
    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    U[-1, :] = np.array([0, np.sin(2*np.pi*1/10), np.sin(2*np.pi*2/10), np.sin(2*np.pi*3/10), 
            np.sin(2*np.pi*4/10), np.sin(2*np.pi*5/10), np.sin(2*np.pi*6/10),  np.sin(2*np.pi*7/10), 
                np.sin(2*np.pi*8/10), np.sin(2*np.pi*9/10), 0]) 
    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/9)):
        U[1, i+1] = Usol[i]
        U[2, i+1] = Usol[i+9]
        U[3, i+1] = Usol[i+18]
        U[4, i+1] = Usol[i+27]
        U[5, i+1] = Usol[i+36]
        U[6, i+1] = Usol[i+45]
        U[7, i+1] = Usol[i+54]
        U[8, i+1] = Usol[i+63]
        U[9, i+1] = Usol[i+72]

    return U, xv, yv

U, xv, yv = num_solution_M9()

def plot3d_sol(U, xv, yv, save = False, savename = "test"):
    # Taken from three_dim_plot (could be added to utilities or something in the end).
    fig = plt.figure()
    fig.suptitle("Num Sol, M = "+str(U.shape[0]-2))
    ax = fig.gca(projection="3d")
    ax.view_init(azim=-45, elev=25) # Added some rotation to the figure. 
    ax.plot_surface(xv, yv, U, cmap="seismic")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("Intensity")
    if save:
        plt.savefig(savename+".pdf")
    plt.show()

#plot3d_sol(U, xv, yv, save=True, savename = "NumSolM9Task3")
