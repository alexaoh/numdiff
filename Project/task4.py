"""Numerical solution of the linearized KdV equation with periodic boundary condition on x \in [-1,1]."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analytical_solution(x, t):
    """Analytical solution of the given KdV equation."""
    return np.sin(np.pi*(x-t))

def plot_analytical_solution():
    """PLot analytical solution to get a feel for how it looks."""
    x = np.linspace(-1, 1, 100)
    t = np.linspace(0, 3, 1000)

    xv, tv = np.meshgrid(x, t)
    U = analytical_solution(xv, tv)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.view_init(azim=55, elev=15) # Added some rotation to the figure. 
    surface = ax.plot_surface(xv, tv, U, cmap="seismic") 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("Intensity")
    fig.colorbar(surface, shrink=0.5, aspect=5)
    plt.show()

def euler_kdv(V, x, t, M, N):
    """Discretization of KdV based on Euler, as derived in report.
    
    V: Grid.
    M: Number of internal points in x-dimension in the grid. 
    N: Number of internal points in t-direction in the grid.  

    Returns a new list X with the solution of the problem in the gridpoints. 
    """
    X = V.copy()

    h = 1/(M+1)
    k = t[1]-t[0]
    r = k/h**2

    # Insert initial condition. 
    X[:, 0] = np.sin(np.pi*x)

    for n in range(N):
        # Vet ikke hva BC-ene er?
        #X[0, n+1] = g0(t[n+1])
        #X[M+1, n+1] = g1(t[n+1])

        # Vet heller ikke helt hvordan man skal starte iterasjonene for de første verdiene!?
        X[1:-1,n+1] = X[1:-1, n] + r/8*(X[3:, n] + 5*X[1:, n] - 16*X[1:-1, n] + 11*X[:-2, n] - X[:-4, n])
    return X

# Test Euler below.
M = 10
N = 10
U = np.zeros((M+2, N+1))
x = np.linspace(-1, 1, M+2)
t = np.linspace(0, 1, N+1)
#euler_kdv(U, x, t, M, N) # Dimensjonsfeil!

def crank_kdv(V, x, t, M, N, u0):
    """Discretization of KdV based on Crank-Nicolson, as derived in report.
    
    V: Grid.
    M: Number of internal points in x-dimension in the grid. 
    N: Number of internal points in t-direction in the grid.  
    u0: Initial condition.
    
    Returns a new list X with the solution of the problem in the gridpoints. 
    """
    X = V.copy()

    h = 1/(M+1)
    k = t[1]-t[0]
    r = k/h**2
    # Continue later. 

plot_analytical_solution()