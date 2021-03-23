import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags # Make sparse matrices with scipy.
from scipy.sparse.linalg import spsolve #Solve linear system with sparse matrix. 
from scipy import sparse # Import sparse for more functions. 
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D


def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2)
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def e_l(U, u):
    """Relative error e_l.

    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    return disc_l2_norm(u-U)/disc_l2_norm(u)

def initial(x):
    """Initial value."""
    return np.sin(np.pi*x)

def anal_solution(x,t):
    """Analytical solution to the problem."""
    return np.sin(np.pi*(x-t))

def theta_method_kdv(x,t,theta):
    """Solves the KdV equation."""
    M = len(x) - 1
    N = len(t) - 1
    U = np.zeros((N+1,M+1))
    U[0,:] = initial(x)

    h = x[1] - x[0]
    k = t[1] - t[0]
    r = k/h**2
    print(r)

    c = 1 + np.pi**2
    a = 1/(8*h**3)
    b = c/(2*h) - 3/(8*h**3)
    data = np.array([np.full(M, a)[:-3], np.full(M, b)[:-1], np.full(M, -b)[:-1], np.full(M,-a)[:-3]])
    diag = np.array([-3, -1, 1, 3])
    Q = diags(data, diag, format = "csc")
    
    Q[0,-3] = Q[1,-2] = Q[2,-1] = a  # Periodic boundary conditions. 
    Q[0,-1] = b
    Q[-1,2] = Q[-2,1] = Q[-3,0] = -a
    Q[-1,0] = -b

    for n in range(N):
        lhs = np.identity(M) - theta*k*Q
        lhs = sparse.csc_matrix(lhs)
        rhs = (np.identity(M) + (1-theta)*k*Q) @ U[n,:-1]
        U[n+1,:-1] = spsolve(lhs,rhs.T) # Trying a sparse solver, but the dimensions do not match anymore!
        # I cannot seem to find out why the dimensions of the rhs change to the tranpose!? (when needing to transpose rhs, this solution is not faster than la.solve!)
        #
    U[:,0] = U[:,-1]
    return U

M = 1000
N = 1000
x = np.linspace(-1, 1, M+1)
t = np.linspace(0, 1, N+1)
U = theta_method_kdv(x, t, 0.5) # Works with theta=1/2 --> Crank Nicolson. 

u = anal_solution(x,t[-1])
#print(U[-1])

'''
plt.plot(x,u,label='an')
plt.plot(x,U[-1],label='num')
plt.legend()
plt.show()
'''

'''
Mlist = np.array([16,32,64,128,256])
disc_err = np.zeros(len(Mlist))
for i, m in enumerate(Mlist):
x = np.linspace(-1,1,m+1)
u = anal_solution(x,t[-1])
U = theta_method_kdv(x,t,0)
disc_err[i] = e_l(U[-1,:],u)

print(disc_err)
h = 1/(Mlist+1)
plt.plot(Mlist,disc_err,label='err')
plt.plot(Mlist,h)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
'''

xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.view_init(azim=55, elev=15) # Added some rotation to the figure. 
surface = ax.plot_surface(xv, tv, U, cmap="seismic") 
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("Intensity")
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.show()
