"""Implementation of Task 2f) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from integrators import RK4_step, RKN34_step
from utilities import plot_order
from scipy.interpolate import interp1d 
from plotting_utilities import plot3d_sol_time
import time


def num_solution(M, N, method):
    """RK4 or RKN34 solution, solved with 'method' step.
    
    Input parameters:
    M: measure of number of grid points along x-direction
    N: measure of number of grid points along t-direction
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
    
    u_0 = lambda x : np.sin(np.pi*x)**2*np.exp(-x**2)
    u_1 = lambda x : np.sin(np.pi*x)**4*np.exp(-x**2)

    x = np.linspace(-2,2,M+2)
    t_i = 1 # the F-function is not dependent of t, set t equal to 1 (random value).
    h = 4/(M+1)
    k = 4/N

    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])/h**2
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M)
    
    F = None
    if method==RK4_step:
        F = lambda t, y : np.array([y[1], Ah.dot(y[0]) - np.sin(y[0])])
    elif method == RKN34_step:
        F = lambda t, v : Ah.dot(v) - np.sin(v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t_i, Y[i,:,:], F)
    
    # Insert B.C.
    U = Y[:,0,:]
    zeros = np.zeros(N+1)
    U = np.insert(U,0,zeros,axis=1)
    U = np.column_stack((U,zeros))
    
    U_der = Y[:,1,:]
    U_der = np.insert(U_der,0,zeros,axis=1)
    U_der = np.column_stack((U_der,zeros))

    return U, U_der

def gauss(deg, f, a, b):
    """Perform Gaussian quadrature on f over the interval [a,b] with deg sample points/weights."""
    [x, w] = np.polynomial.legendre.leggauss(deg)
    Q = 0.5*(b - a)*sum(w*f(0.5*(b - a)*x + 0.5*(b + a)))
    return Q

def calc_E(x,u,u_t):
    M = len(x) - 2
    h = x[1] - x[0]
    
    data = np.array([np.full(M+2, -1), np.full(M+2, 1)]) # Perhaps it would be better to differentiate manually with slicing of arrays?
    diags = np.array([-1, 1])
    B = spdiags(data, diags, M+2, M+2,format='lil')
    boundary = np.array([3,-4,1])
    B[0,:3] = -boundary; B[-1,-3:] = np.flip(boundary)
    B = B.tocsr()
    
    u_x = B.dot(u)/(2*h)
    E_x_list = (1/2)*(u_t**2 + u_x**2) + 1 - np.cos(u)
    interp_E_x = interp1d(x,E_x_list,kind='cubic')

    return gauss(2800,interp_E_x,x[0],x[-1]) 
    
def plot_energy(x,u,u_t,savename=False):
    E = np.zeros(N+1)
    for i in range(N+1):
        E[i] = calc_E(x,u[i],u_t[i])

    plt.plot(t,E,color='royalblue',label='$E(t)$')
    plt.xlabel('$t$')
    plt.ylabel('Energy')
    plt.legend(loc='upper center')
    if savename:
        plt.savefig(savename + '.pdf')
    plt.show()
    
def plot_sol(x,U,savename=False):
    plt.plot(x,U[0],color='royalblue',label=r'$u(x,0)$')
    plt.plot(x,U[-1],color='limegreen',label=r'$u(x,4)$')
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')
    plt.legend()
    if savename:
        plt.savefig(savename+'.pdf')
    plt.show()
    
def energy_refinement(M, N, solvers, plot = False, savename = False):
    assert(isinstance(solvers,list))
    
    if np.ndim(M) == 0:
        M = np.ones_like(N)*M
    elif np.ndim(N) == 0:
        N = np.ones_like(M)*N
    else:
        assert(len(M)==len(N))

    energy_diff = np.zeros((len(solvers),len(M)))
    for i, method in enumerate(solvers):
        for j, m in enumerate(M):
            x = np.linspace(-2,2,m+2)
            U, U_t = num_solution(m,N[j],method)
            E_0 = calc_E(x,U[0],U_t[0])
            E_end = calc_E(x,U[-1],U_t[-1])
            energy_diff[i,j] = np.abs(E_end-E_0)/np.abs(E_0)

    Ndof = M*N
    if plot:
        if len(solvers)==2:
            plt.plot(Ndof, energy_diff[0,:], label=r"$\Delta E$ (RK4)", color='red', marker = 'o')
            plt.plot(Ndof, energy_diff[1,:], label=r"$\Delta E$ (RKN34)", color='blue', marker = 'o')
        else:
            plt.plot(Ndof, energy_diff[0,:], label=r"$\Delta E$", color='red', marker = 'o')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$M \cdot N$")
        plt.ylabel(r"$\Delta E$")
        plt.legend()
        plt.grid()
        if savename:
            plt.savefig(savename+".pdf")
        plt.show()

def comp_time(M,N,solvers,savename=False):
    """Calculate the elapsed time when using the methods RK4 and RKN34 and plots the time."""
    assert(isinstance(solvers,list))
    
    if np.ndim(M) == 0:
        M = np.ones_like(N)*M
    elif np.ndim(N) == 0:
        N = np.ones_like(M)*N
    else:
        assert(len(M)==len(N))
    
    time_elapsed = np.zeros((len(solvers),len(M)))
    for i, method in enumerate(solvers):
        for j, m in enumerate(M):
            x = np.linspace(-2,2,m+2)
            time_start = time.time()
            U, U_t = num_solution(m,N[j],method)
            time_elapsed[i,j] = time.time() - time_start
    
    Ndof = M*N
    plt.plot(Ndof, time_elapsed[0], label = "RK4", color='red',marker='o')
    plt.plot(Ndof, time_elapsed[1], label = "RKN34", color='blue',marker='o')
    
    plt.xscale('log')
    plt.xlabel(r"$M \cdot N$")
    plt.ylabel("time (seconds)")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename +'.pdf')
    plt.show()

# ===| Run code below.|=== #

# Plot of numerical solution and energy
N = 500; M = 350; T = 4
x = np.linspace(-2,2,M+2)
t = np.linspace(0,T,N+1)
U,U_der = num_solution(M, N, RK4_step)

#plot_sol(x,U)
#plot3d_sol_time(U,x,t,110,20)
#plot_energy(x,U,U_der) #Obs, takes long time; change deg in Gauss first

# ---Energy refinement k=ch---
M = 2**np.arange(5,13)
solvers = [RK4_step, RKN34_step]

N = 1.5*M
N = np.array(N,dtype=int)
#energy_refinement(M, N, solvers, plot = True)

N = 4*M
N = np.array(N,dtype=int)
#energy_refinement(M, N, solvers, plot = True)

# ---Compute Time Spent---
N = 20000
#comp_time(M,N,solvers)
