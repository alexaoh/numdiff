# Try with Crank Nicholson perhaps. 
from crank_nicolson import *
from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np

initial = (lambda x: 2*np.pi*x - np.sin(2*np.pi*x))

M = 4 # Internal points in x dimension.
N = 500 # Internal points in t dimension.
L = 1 # Length of rod. 
x = np.linspace(0, L, M+1) # x-axis.
t = np.linspace(0, 0.5, N+1) # t-axis. 
h = x[1]-x[0]

V = np.zeros((M+1, M+1)) # Make grid. 
V[:,0] = initial(x) # Add initial condition. 

data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
diags = np.array([0, 1, 2])
Q = spdiags(data, diags, M+1, M+1).toarray()

# Solution of the problem with Crank-Nicolson.
V = trapezoidal_method(V, Q, t, h)

# Visualize in 3d and with subplots for some times. 
# tv, xv = np.meshgrid(t,x)
# three_dim_plot(xv = xv, tv = tv, I = U, label = "Numerical Solution")
# sub(x = x, I = U, t = t, L = L, t_index = [0, int(N/500), int(N/70), int(N/10)], label = "Numerical solution at some times")

# Need to draw the convergence plot with CN and Theta-method. 
# How is this done/what is the convergence plot?
