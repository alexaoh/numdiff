# Try with Crank Nicholson perhaps. 
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
# V = trapezoidal_method(V, Q, t, h)

# Visualize in 3d and with subplots for some times. 
# tv, xv = np.meshgrid(t,x)
# three_dim_plot(xv = xv, tv = tv, I = U, label = "Numerical Solution")
# sub(x = x, I = U, t = t, L = L, t_index = [0, int(N/500), int(N/70), int(N/10)], label = "Numerical solution at some times")

# Need to draw the convergence plot with CN and Theta-method. 
# How is this done/what is the convergence plot?

def secondOrder(M, plot = True):
    ''' Use central differences + trapezoidal '''

    
    N = 10 # Internal points in t dimension.
    L = 1 # Length of rod. 

    # Construct Q
    data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
    diags = np.array([-1, 0, 1])
    Q = spdiags(data, diags, M+1, M+1).toarray()
    # Q[-1, -2] = Q[0, 1] = 2 
    Q[0, 0] = Q[-1, -1] = -1
    Q[0, 1] = Q[-1, -2] = 0
    Q[0, 2] = Q[-1,-3] = 1
    #print(Q)

    xGrid = np.linspace(0, L, M + 1)
    h = xGrid[1]-xGrid[0]
    tGrid = np.linspace(0, 0.5, N+1) # t-axis. 
    V0 = [initial(x) for x in xGrid]

    sol = trapezoidal_method(V0, Q, t, h)
    #print("sol:")
    #print(sol)

    # Plotting to check, looks promising.
    # Maybe alex has some fancy plotting methods?
    #plt.plot(xGrid,sol[0,:])
    #plt.plot(xGrid,sol[5,:])
    #plt.plot(xGrid,sol[10,:])
    #plt.plot(xGrid, sol[20,:])
    #plt.plot(xGrid, sol[40,:])
    #plt.plot(xGrid, sol[-1,:])

    #plt.show()
    if plot:
        tv, xv = np.meshgrid(tGrid,xGrid)
        three_dim_plot(xv = xv, tv = tv, I = sol.T, label = "Numerical Solution")
    
    return sol


def saveRefSol(Mstar, filename):
    '''Saves the reference solution to file.'''
    refSol = secondOrder(Mstar, plot = False)
    with open(filename, 'wb') as fi:
        pickle.dump(refSol, fi)



def calcError(Mstar, filename):
    '''Calculates the relative error with the reference solution'''
    refSol = None
    with open(filename, 'rb') as fi:
        refSol = pickle.load(fi)

    Mlist = [i for i in range(10, 200)]
    errorList = []
    piecewise = lambda u, refSol, M, Mstar : [refSol[math.floor(i*Mstar/M)] for i in range(len(u))]
    for M in Mlist:
        sol = secondOrder(M, plot = False)
        u = sol[5,:]
        uStar = np.array(piecewise(u, refSol[5,:], M, Mstar))
        #print("Ustar")
        #print(uStar)
        #print(sol)
        err = e_l(u, uStar)
        errorList.append(err)
    
    plt.plot(Mlist, errorList)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


Mstar = 10000
filename = 'refSol.pk'

saveRefSol(Mstar, filename) # Only needs to be run once, or if you change Mstar
calcError(Mstar, filename)


