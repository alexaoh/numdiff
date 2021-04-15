"""old code for task 4, didn't want to delete it quite yet."""
import numpy as np
from scipy.sparse import spdiags
import numpy.linalg as la


def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2)
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def theta_method_kdv(x,t,theta,init):
    """Solves the KdV equation.
    x,t: grids to solve on
    theta: parameter to choose method, 0 for forward Euler and 1/2 for CN
    init: function setting the initial condition u(x,0)
    """
    M = len(x) - 1
    N = len(t) - 1
    U = np.zeros((N+1,M+1))
    U[0,:] = init(x)

    h = x[1] - x[0]
    k = t[1] - t[0]

    c = 1 + np.pi**2
    a = 1/(8*h**3)
    b = c/(2*h) - 3/(8*h**3)
    data = np.array([np.full(M+1, a), np.full(M+1, b), np.full(M+1, -b), np.full(M+1,-a)])
    diags = np.array([-3, -1, 1, 3])
    Q = spdiags(data, diags, M+1, M+1).toarray()

    #periodic BC
    Q[0,-4] = Q[1,-3] = Q[2,-2] = a  
    Q[0,-2] = b
    Q[-1,3] = Q[-2,2] = Q[-3,1] = -a
    Q[-1,1] = -b

    lhs = np.identity(M+1) - theta*k*Q
    b = (np.identity(M+1) + (1-theta)*k*Q)
    for n in range(N):
        rhs = b @ U[n,:]
        U[n+1,:] = la.solve(lhs,rhs) # Could try a sparse solver eventually also, if this is too slow or too memory-demanding. 
    return U