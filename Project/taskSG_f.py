"""Implementation of Task 2f) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import spdiags
from scipy.integrate import simps
from integrators import RK4_step, RKN34_step

u_0 = lambda x : np.sin(np.pi*x)**2*np.exp(-x**2)
u_1 = lambda x : np.sin(np.pi*x)**4*np.exp(-x**2)

def numeric_solution(x, t, method):
    """RK4 or RKN34 solution, solved with 'method' step.
    
    Input parameters:
    t: time-grid
    x: x-grid
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
       
    N = len(t)-1
    M = len(x)-2
    k = t[1] - t[0]
    h = x[1] - x[0]
    
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    g = lambda t, v : np.concatenate(([-np.sin(v[0])],-np.sin(v[1:-1]),[-np.sin(v[-1])])) #zero as B.C
    F = None
    if method==RK4_step:
        F = lambda t, y : np.array([y[1], Ah @ y[0] + g(t,y[0])])
    elif method == RKN34_step:
        F = lambda t, v : Ah @ v + g(t,v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))
    #Initial conditions
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t[i], Y[i,:,:], F)
    
    Usol = Y[:,0,:]
    
    #Insert B.C
    Usol = np.insert(Usol,0,np.zeros_like(t),axis=1)
    Usol = np.column_stack((Usol,np.zeros_like(t)))

    return Usol

N = 400
M = int(0.7*N)
T = 4

x = np.linspace(-2,2,M+2)
t = np.linspace(0,T,N+1)

U = numeric_solution(x, t, RK4_step)
V = numeric_solution(x, t, RKN34_step)

plt.plot(x,U[-1,:], label = "RK4")
plt.plot(x,V[-1,:],label='RKN34')
plt.legend()
plt.show()

def energy(x,t,U,i):
    """Calculates energy for a given time t[i]"""
    u_t, u_x = np.gradient(U,t,x,edge_order=2)
    y = 0.5*u_t[i,:]**2 + 0.5*u_x[i,:]**2 + np.cos(U[i,:])
    E = simps(y,x)
    return E 

def energy_0():
    u_0_x = lambda x : np.exp(-x**2)*(np.pi*np.sin(2*np.pi*x) - 2*x*np.sin(np.pi*x)**2)  #= u_x(x,0)
    M = 1000
    x = np.linspace(-2,2,M+2)
    y = 0.5*u_1(x)**2 + 0.5*u_0_x(x)**2 + np.cos(u_0(x))
    return simps(y,x)

#Are we supposed to do some sort of refinement here? For instance h,t-refinement.
def energy_refinement(method):
    #N = np.array([40,80,160,320])
    N = np.linspace(100,600,30,dtype=int)  
    M = N*0.7
    M = M.astype(int)
    energy_diff = np.zeros(len(N))
    E_0 = energy_0()
    for i in range(len(N)):
        x = np.linspace(-2,2,M[i]+2)
        t = np.linspace(0,4,N[i]+1)
        U = numeric_solution(x,t,method)
        energy_diff[i] = np.abs(energy(x,t,U,-1)-E_0)/E_0
    
    plt.plot(M*N,energy_diff)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#energy_refinement(RK4_step)