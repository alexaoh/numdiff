"""Numerical solution using the given difference method in task 1a."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def anal_solution(x, alpha, sigma):
    """Analytical solution of the Possion equation with given Neumann BC."""
    return -1/(4*np.pi**2)*np.cos(2*np.pi*x) + 1/6*x**3 + (sigma - 1/2)*x + alpha + 1/(4*np.pi**2)

M = 20
h = 1/(M+1)

# Construct Ah. 
data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
diags = np.array([-1, 0, 1])
Ah = spdiags(data, diags, M+1, M+1).toarray()*1/h**2
Ah[-1, -3] = -h/2
Ah[-1, -2] = 2*h
Ah[-1, -1] = -3*h/2

x = np.linspace(0, 1, M+2) # Make 1D grid.
U = np.zeros_like(x[1:]) # Array to hold approximations in grid points. 
# This U has not been used, since I used la.solve below instead. 

def f(x):
    return np.cos(2*np.pi*x) + x

alpha = 0
sigma = 0

# Construct f.
f_vec = np.full(M+1, f(x[1:]))
f_vec[0] = f_vec[0] - alpha/h**2
f_vec[-1] = sigma

Usol = la.solve(Ah, f_vec)

# Add left Dirichlet condition to solution.
Usol = np.insert(Usol, 0, alpha)

# Plot solution to have a look. 
plt.plot(x, Usol, label="Num", color = "red")
plt.plot(x, anal_solution(x, alpha, sigma), label="An", color = "black", linestyle = "dotted")
plt.legend()
plt.show()
# Obviously something wrong here. Check later. 
