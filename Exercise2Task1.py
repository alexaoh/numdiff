"""Solve the given version of the heat equation on x \in [0,1]"""
from crank_nicolson import *
from theta_method import *
from plot_heat_eqn import *

def f(x):
    """Initial condition in task a)"""
    y = x.copy()
    for i in range(len(y)):
        if 0 <= y[i] <= 0.5:
            y[i] = 2*y[i]
        elif  0.5 < y[i] <= 1:
            y[i] = 2-2*y[i]
        else: 
            raise Exception("Something wrong in x!")
    return y

def fb(x):
    """Discontinuous initial condition from task b).
    
    The smoothing property of the heat equation is seen when plotting. 
    """
    y = x.copy()
    for i in range(len(y)):
        if 0 <= y[i] <= 0.3:
            y[i] = y[i]
        elif  0.3 < y[i] <= 0.6:
            y[i] = 0.6
        elif 0.6 <= y[i] <= 1:
            y[i] = 1
        else: 
            raise Exception("Something wrong in x!")
    return y
    
g0 = g1 = (lambda x: 0) # Boundary conditions. 
M = 71 # Internal points in x dimension.
N = 500 # Internal points in t dimension.
L = 1 # Length of rod. 
x = np.linspace(0, L, M+2) # x-axis.
t = np.linspace(0, 0.5, N+1) # t-axis. 


V = np.zeros((M+2, N+1)) # Make grid. 
V[:,0] = fb(x) # Add initial condition. 

# Solution of the problem with Crank-Nicolson.
V = crank_nicolson(V, x, t, M, N, g0, g1)

U = theta_method(V, x, t, M, N, g0, g1, theta = 0.8)

# Visualize in 3d and with subplots for some times. 
tv, xv = np.meshgrid(t,x)
three_dim_plot(xv = xv, tv = tv, I = U, label = "Numerical Solution")
sub(x = x, I = U, t = t, L = L, t_index = [0, int(N/500), int(N/70), int(N/10)], label = "Numerical solution at some times")

# Need to draw the convergence plot with CN and Theta-method. 
# How is this done/what is the convergence plot?
