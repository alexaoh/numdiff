import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags # Make sparse matrices with scipy.
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

def initial_sine(x):
    return np.sin(np.pi*x)

def anal_solution(x,t):
    return np.sin(np.pi*(x-t))

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

'''
#Code for 3D plot, should we include this in the rapport?
M = 30
N = 1000
x = np.linspace(-1, 1, M+1)
t = np.linspace(0, 1, N+1)  #Show error at time t=1.
U = theta_method_kdv(x, t, 0.5,initial_sine) #CN with theta=1/2

u = anal_solution(x,t[-1])

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
'''

def plot_disc_error_at_tmax(M,theta,savefig=False):
    """Plot l_2 norm at t=1 as a function discretization number M along x. For task 4b)."""
    t_max = 1
    N = 1000
    t = np.linspace(0,t_max,N+1)
    disc_err = np.zeros(len(M))
    for i, m in enumerate(M):
        x = np.linspace(-1,1,m+1)
        u = anal_solution(x,t[-1])
        U = theta_method_kdv(x,t,theta,initial_sine)
        disc_err[i] = e_l(U[-1,:],u)

    Ndof = M*N
    h = 1/Ndof
    plt.plot(Ndof,disc_err,label=r"$e^r_{\ell}$ CN",color="red",marker='o')
    plt.plot(Ndof,(3e+8)*h**2,label=r"O$(N_{dof}^{-2})$",color="red",linestyle='dashed')
    
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel(r"$M\cdot N$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    if savefig:
        plt.savefig('task4b_error.pdf')
    plt.show()

#M = np.array([8,16,32,64,128,256,512])
#plot_disc_error_at_tmax(M,1/2)  #using Crank-Nicolson
#plot_disc_error_at_max(M,0)  #solution blows up using Euler - unstable. The labels are not correct for this plot.


#-----Code for 4c -----

def plot_l2_norm(x,t,theta,init,skip):
    """Plot l2 norm as a function of time.
    
    theta: parameter choosing method, CN = 1/2
    init: function to set initial condintions
    skip: plot norm at every 'skip' t-value
    """
    U = theta_method_kdv(x, t, theta, init)
    t_skipped = t[::skip]
    indices = [i for i in range(0,len(t),skip)]
    l2_norm = np.zeros(len(t_skipped))
    for i in range(len(indices)):
        l2_norm[i] = disc_l2_norm(U[indices[i],:])

    plt.plot(t_skipped, l2_norm, label=r"$||v||$", color='red')
    plt.ylim(0.704,0.71)
    plt.ylabel(r"$\ell_2$")
    plt.xlabel("$t$")
    plt.legend()
    #plt.savefig('task4c_sine.pdf')
    plt.show()

#Here using Crank-Nicolson (theta=1/2)
M = 800
N = 1000
x = np.linspace(-1, 1, M+1)
t = np.linspace(0, 10, N+1)
#plot_l2_norm(x,t,1/2,initial_sine,5)


#Choose our own initial condition
initial_sine_2 = lambda x : np.sin(2*np.pi*x)
#plot_l2_norm(x,t,1/2,initial_sine_2,5)


#To show implementation of Forward Euler. Solution blows up at once, cannot plot l2 norm.
#plot_l2_norm(x,t,0,initial_sine,5)  
#plot_l2_norm(x,t,0,initial_sine_2,5)