
'''
Solves u_t = u_xx on x[0,1], t[0,T] with reference to manufactured solution, both with UMR and AMR and both first and second order
Here we have Dirichlet BC u(0,t)=u(1,t)=0, and initial value f(x)=3*sin(2*pi*x)
First order method; uses Eulers method (forward)
Second order; Crank Nicolson
'''

from crank_nicolson import *
from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import math
import pickle # To save the reference solution.

def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2) 
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def e_l(U, u):
    """Relative error e_l.

    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    return disc_l2_norm(u-U)/disc_l2_norm(u) # la.norm(u-U)/la.norm(u)

initial = (lambda x: 3*np.sin(2*np.pi*x))

def anal_solution(x,t):
    return 3*np.exp(-4*(np.pi**2)*t)*np.sin(2*np.pi*x)

def Eulers_method(x,t,M,N):
    r = (t[1]-t[0])/(x[1]-x[0])**2  #k/h^2 = (M+2)^2/(N+1), must have r <= 0.5 for convergence
    print(r)
    U = np.zeros((N+1,M+2))
    U[0,:] = initial(x)
    
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    A = spdiags(data, diags, M, M).toarray()
   
    for n in range(N):
        U[n+1,1:-1] = U[n,1:-1] + r* (A @ U[n,1:-1])
    return U

M = 15
x = np.linspace(0,1,M+2)
N = 100
T = 0.5  #arbitrary value
t = np.linspace(0,T,N+1)
U = Eulers_method(x,t,M,N)


plt.plot(x,U[4,:])
plt.show()

"""
def calcSol(M, order, plot = True):
    ''' 
    order = 1: Use bd and fd on bc + trapezoidal.
    order = 2: Use central differences with fict. nodes on bc + trapezoidal.
    '''
    
    N = 50 # Internal points in t dimension.
    L = 1 # Length of rod. 

    # Construct Q
    data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
    diags = np.array([-1, 0, 1])
    Q = spdiags(data, diags, M+1, M+1).toarray()
    if order == 1:
        Q[0, 0] = Q[-1, -1] = -1
        Q[0, 1] = Q[-1, -2] =  0
        Q[0, 2] = Q[-1,-3] = 1

    elif order == 2:
        Q[0, 1] = Q[-1, -2] =  2 
    
    
    xGrid = np.linspace(0, L, M + 1)
    h = xGrid[1]-xGrid[0]
    tGrid = np.linspace(0, 0.2, N+1) # t-axis. 
    V0 = [initial(x) for x in xGrid]
    
    sol = trapezoidal_method(V0, Q, tGrid, h)
  
    if plot:
        tv, xv = np.meshgrid(tGrid,xGrid)
        three_dim_plot(xv = xv, tv = tv, I = sol.T, label = "Numerical Solution")
    
    return sol
"""



