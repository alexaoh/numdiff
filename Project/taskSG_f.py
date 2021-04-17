"""Implementation of Task 2f) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import spdiags
from integrators import RK4_step, RKN34_step
from scipy.integrate import quad, quadrature
from scipy.interpolate import interp1d 


u_0 = lambda x : np.sin(np.pi*x)**2*np.exp(-x**2)
u_1 = lambda x : np.sin(np.pi*x)**4*np.exp(-x**2)

def numeric_solution(M, N, method):
    """RK4 or RKN34 solution, solved with 'method' step.
    
    Input parameters:
    t: time-grid
    x: x-grid
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
       
    #N = len(t)-1
    #M = len(x)-2
    x = np.linspace(-2,2,M+2)
    #t = np.linspace(0,4,N+1)
    t_i = 1 #the F-function is not dependent upon t, set t equal to 1 (random value)
    h = 4/(M+1)
    k = 4/N
    #k = 4/N
    #h = 4/(M+1)
    
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    #g = lambda v : np.concatenate(([-np.sin(v[0])],-np.sin(v[1:-1]),[-np.sin(v[-1])])) #zero as B.C
    F = None
    if method==RK4_step:
        F = lambda t, y : np.array([y[1], Ah @ y[0] - np.sin(y[0])])
    elif method == RKN34_step:
        F = lambda t, v : Ah @ v - np.sin(v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))  
    #Initial conditions
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t_i, Y[i,:,:], F)
    
    
    U = Y[-1,:,:]
    zeros = np.zeros(2)
    U = np.insert(U,0,zeros,axis=1)
    U = np.column_stack((U,zeros))
    
    #Insert B.C
    #zeros = np.full(N+1,0)
    #Usol = np.insert(Usol,0,zeros,axis=1)
    #Usol = np.column_stack((Usol,zeros))

    return U #Y[-1,:,:] #Only need last time-step

def calc_E(x,u,u_t):
    h = x[1] - x[0]
    M = len(x)-2
    
    data = np.array([np.full(M+2, -1), np.full(M+2, 1)])
    diags = np.array([-1, 1])
    Bh = spdiags(data, diags, M+2, M+2).toarray()/(2*h)
    boundary = np.array([3,-4,1])/(2*h)
    Bh[0,:3] = -boundary; Bh[-1,-3:] = np.flip(boundary)
    
    """Alt. calculation of u_x; (more compact, maybe faster than making the matrix for then to do matrix mult.)
    U_x = np.zeros(M+2)
    U_x[0] = (-3*U[0,0] + 4*U[0,1] - U[0,2])/(2*h)
    U_x[1:-1] = (U[0,2:]-U[0,:-2])/(2*h)
    U_x[-1] = (U[0,-3]-4*U[0,-2] + 3*U[0,-1])/(2*h)"""
    
    E_x_list = -(1/2)*(u_t**2 + (Bh @ u)**2) + np.cos(u)
    interp_E_x = interp1d(x,E_x_list,kind='cubic')
    return quad(interp_E_x,x[0],x[-1],epsabs=2e-6)[0]

N = 400
M = int(0.7*N)
#T = 4
#x = np.linspace(-2,2,M+2)
#t = np.linspace(0,T,N+1)
x = np.linspace(-2,2,M+2)

U = numeric_solution(M, N, RK4_step)
#V = numeric_solution(M, N, RKN34_step)

#plt.plot(x,U[0], label = "RK4")
#plt.legend()
#plt.show()

def plot_order(Ndof, error_start, order, label, color):
    """Plots Ndof^{-order} from a starting point."""
    const = (error_start)**(1/order)*Ndof[0]
    plt.plot(Ndof, (const*1/Ndof)**order, label=label, color=color, linestyle='dashed')
    
#What type of refinement are we supposed to do?
def energy_refinement(method):
    M = np.array([40,80,160,320,500])
    N = 1200
    #N = np.linspace(100,600,30,dtype=int)  
    #M = N*0.7
    #M = M.astype(int)
    energy_diff = np.zeros(len(M))
    for i in range(len(M)):
        x = np.linspace(-2,2,M[i]+2)
        #t = np.linspace(0,4,N[i]+1)
        U = numeric_solution(M[i],N,method)
        E_0 = calc_E(x,u_0(x),u_1(x))
        E_end = calc_E(x,U[0],U[1])
        energy_diff[i] = np.abs(E_end - E_0)/E_0
    
    Ndof = (M*N)
    plt.plot(Ndof,energy_diff)
    plot_order(Ndof,energy_diff[0],2,"O(h^2)",'red')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#energy_refinement(RK4_step)