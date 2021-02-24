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



def calcSol(M, order, plot = True):
    ''' 
    order = 1: Use bd and fd on bc + trapezoidal.
    order = 2: Use central differences with fict. nodes on bc + trapezoidal.
    '''
    
    N = 1000 # Internal points in t dimension.
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
    print(Q)
    

    xGrid = np.linspace(0, L, M + 1)
    h = xGrid[1]-xGrid[0]
    tGrid = np.linspace(0, 0.2, N+1) # t-axis. 
    V0 = [initial(x) for x in xGrid]
    
    sol = trapezoidal_method(V0, Q, tGrid, h)
  
    if plot:
        tv, xv = np.meshgrid(tGrid,xGrid)
        three_dim_plot(xv = xv, tv = tv, I = sol.T, label = "Numerical Solution")
    
    return sol


def saveRefSol(Mstar, order, filename):
    '''Saves the reference solution to file.'''
    refSol = calcSol(Mstar, order, plot = False)
    with open(filename, 'wb') as fi:
        pickle.dump(refSol, fi)



def calcError(Mstar, order, filename):
    '''Calculates the relative error with the reference solution'''
    refSol = None
    refTime = -1 # Time step at which to calculate error.
    with open(filename, 'rb') as fi: # Fetches the calculated reference solution
        refSol = pickle.load(fi)

    Mlist = [i for i in range(5, 100)]
    Mlist += [200,300,400,500,700,900,1000]
    errorList = []
    hList = []
    hList2 = []

    # Piecewise constant function uStar
    piecewise = lambda Ustar, M, Mstar : [Ustar[math.floor(i*(Mstar + 1)/(M + 1))] for i in range(M + 1)]

    for M in Mlist:
        sol = calcSol(M, order, plot = False)
        u = sol[refTime,:]
        uStar = np.array(piecewise(refSol[refTime,:], M, Mstar))
        err = e_l(u, uStar)

        errorList.append(err)
        hList.append(1/(M + 1))
        hList2.append((1/(M + 1))**2)
           
    
    plt.plot(Mlist, hList2, label="$O(h^2)$", linestyle='dashed') 
    plt.plot(Mlist, hList, label="$O(h)$", linestyle='dashed') 
    plt.plot(Mlist, errorList, label = "$e^r_l$")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.show()


Mstar = 1000
filename = 'refSol.pk'

saveRefSol(Mstar, 2, filename) # Only needs to be run once, or if you change Mstar
calcError(Mstar, 2, filename)




