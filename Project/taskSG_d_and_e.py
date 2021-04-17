"""Implementation of Task 2d,e) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from integrators import RK2_step, RK3_step, RK4_step, RKN12_step, RKN34_step
from utilities import e_l, plot_order
from plotting_utilities import plot3d_sol_part2

c = 0.5  
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the plus-sign as solution."""
    b = (x-c*t)*(1-c**2)**(-1/2)
    return 4*np.arctan(np.exp(b))

def num_solution(x, t, method):
    """Numeric solution using Runge-Kutta or Runge-Kutta-Nystrøm schemes, solved with 'method' step.
    
    Input parameters:
    x: x-grid
    t: time-grid
    method: callable step function for RK or RKN loop. 
    """
    assert(callable(method))
    
    f_1 = lambda t : analytical_solution(x[0],t)
    f_2 = lambda t : analytical_solution(x[-1],t)
    u_0 = lambda x : analytical_solution(x,0)
    u_1 = lambda x : -4*c*np.exp(x*(1-c**2)**(-1/2))/((1-c**2)**(1/2)*(1 + np.exp(2*x*(1-c**2)**(-1/2))))
    
    N = len(t)-1
    M = len(x)-2
    k = t[1] - t[0]
    h = x[1] - x[0]
    
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])/h**2
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M)
    
    g = lambda t, v : np.concatenate(([-np.sin(v[0])+f_1(t)/h**2],-np.sin(v[1:-1]),[-np.sin(v[-1])+f_2(t)/h**2]))
    F = None
    if (method==RK2_step) or (method==RK3_step) or (method==RK4_step):
        F = lambda t, y : np.array([y[1], Ah.dot(y[0]) + g(t,y[0])])
    elif (method==RKN12_step) or (method==RKN34_step):
        F = lambda t, v : Ah.dot(v) + g(t,v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t[i], Y[i,:,:], F)
    
    Usol = Y[:,0,:]
    
    #Insert B.C
    Usol = np.insert(Usol,0,f_1(t),axis=1)
    Usol = np.column_stack((Usol,f_2(t)))

    return Usol

#Løsningen divergerer når N<M, noe som er feil eller er det bare CFL-condition som spiller inn-evt. dispersion som er problemet.
#RK2; M/N <= 0.65
#RK3; M/N <= 1.8
#RK4; M/N <= 2.6
#RKN12; ...
#RKN34; ...


M=20; N=10; T = 5
x = np.linspace(-5,5,M+2)
t = np.linspace(0,T,N+1)
U = num_solution(x, t, RK4_step)
plot3d_sol_part2(x,t,U,analytical_solution)


def refinement(M,N,method,savename=False):
    """Preforms h- or t-refinement and plots the result."""
    T = 5
    if np.ndim(M) == 0:
        M = np.ones_like(N)*M
    elif np.ndim(N) == 0:
        N = np.ones_like(M)*N
    else:
        assert(len(M)==len(N))
        
    err = np.zeros(len(M))
    for i in range(len(M)):
        x = np.linspace(-5,5,M[i]+2)
        t = np.linspace(0,T,N[i]+1) 
        U = num_solution(x,t,method)
        u = analytical_solution(x,t[-1])
        err[i] = e_l(U[-1,:],u)

    Ndof = M*N
    #Maybe we need more plots in the same figure?
    plt.plot(Ndof, err, label=r"$e^r_{\ell}$", color='red', marker = 'o')
    plot_order(Ndof, err[0], 2, label=r"$\mathcal{O}(N_{dof}^{-2})$", color="red")
    plt.suptitle('RK4')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$M \cdot N$")
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()
    
M = np.array([8,16,32,64,128])
N = 800
#refinement(M,N,RK4_step)