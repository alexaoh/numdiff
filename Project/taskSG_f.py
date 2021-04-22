"""Implementation of Task 2f) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from integrators import RK4_step, RKN34_step
from utilities import plot_order
from scipy.integrate import quad, quadrature #what is the diff. between those two?
from scipy.interpolate import interp1d 
from plotting_utilities import plot3d_sol_part2
import time


def num_solution(M, N, method): #May consider generalizing the func. in d&e) to handle input of B.C-functions, but it's less efficient.
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
    #t = np.linspace(0,4,N+1)
    t_i = 1 #the F-function is not dependent upon t, set t equal to 1 (random value)
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
    
    #Insert B.C
    U = Y[:,0,:]
    zeros = np.zeros(N+1)
    U = np.insert(U,0,zeros,axis=1)
    U = np.column_stack((U,zeros))
    
    U_der = Y[:,1,:]
    U_der = np.insert(U_der,0,zeros,axis=1)
    U_der = np.column_stack((U_der,zeros))

    return U, U_der

N = 400
M = int(0.7*N)
T = 4
x = np.linspace(-2,2,M+2)
t = np.linspace(0,4,N+1)
U,_ = num_solution(M, N, RK4_step)
#plot3d_sol_part2(x,t,U)

def calc_E(x,u,u_t):
    M = len(x) - 2
    h = x[1] - x[0]
    
    data = np.array([np.full(M+2, -1), np.full(M+2, 1)]) #maybe it's better differentiate manually with slicing of arrays?
    diags = np.array([-1, 1])
    B = spdiags(data, diags, M+2, M+2,format='lil')
    boundary = np.array([3,-4,1])
    B[0,:3] = -boundary; B[-1,-3:] = np.flip(boundary)
    B = B.tocsr()
    
    u_x = B.dot(u)/(2*h)
    E_x_list = -(1/2)*(u_t**2 + u_x**2) + np.cos(u)
    interp_E_x = interp1d(x,E_x_list,kind='cubic')
    return quad(interp_E_x,x[0],x[-1],epsabs=2e-6)[0] #can we change this error?
  
#What type of refinement are we supposed to do?
def energy_refinement(M, N, method, plot = False, savename = False):

    if np.ndim(M) == 0:
        M = np.ones_like(N)*M
    elif np.ndim(N) == 0:
        N = np.ones_like(M)*N
    else:
        assert(len(M)==len(N))
    
    energy_diff = np.zeros(len(M))
    time_elapsed = np.zeros(len(M)) # For saving the computational times.
    for i in range(len(M)):
        x = np.linspace(-2,2,M[i]+2)
        time_start = time.time()
        U, U_t = num_solution(M[i],N[i],method)
        time_elapsed[i] = time.time() - time_start
        E_0 = calc_E(x,U[0],U_t[0])
        E_end = calc_E(x,U[-1],U_t[-1])
        energy_diff[i] = np.abs(E_end - E_0)/E_0
    


    Ndof = M*N
    if plot:
        #Maybe we need more plots in the same figure?
        plt.plot(Ndof, energy_diff, label=r"$\Delta E$", color='red', marker = 'o')
        plot_order(Ndof, energy_diff[0], 2, label=r"$\mathcal{O}(N_{dof}^{-2})$", color="red")
        plt.suptitle('RK4')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$M \cdot N$")
        plt.ylabel(r"$\Delta E$")
        plt.legend()
        plt.grid()
        if savename:
            plt.savefig(savename+".pdf")
        plt.show()

    return time_elapsed, Ndof


def comp_time(M, N):
    """Calculates the elapsed time when using the methods RK4 and RKN34 and plots the time."""
    time_RK4, Ndof = energy_refinement(M, N, RK4_step)
    time_RKN34, Ndof = energy_refinement(M, N, RKN34_step)
    plt.plot(Ndof, time_RK4, label = "RK4")
    plt.plot(Ndof, time_RKN34, label = "RKN34")

    plt.xlabel("$M \cdot N$")
    plt.ylabel("time (seconds)")
    plt.legend()
    plt.show()


M = np.array([40, 80, 160, 320, 500])
N = 1200
#energy_refinement(M, N, RK4_step, plot = True)
comp_time(M,N)



