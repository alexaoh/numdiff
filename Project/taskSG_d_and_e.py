"""Implementation of Task 2d,e) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import spdiags
from integrators import RK2_step, RK3_step, RK4_step, RKN12_step, RKN34_step

c = 0.5  #Kan vi bestemme c selv?
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the plus-sign as solution."""
    b = (x-c*t)*(1-c**2)**(-1/2) #Valgte plus-tegnet som løsning, det er vel greit?
    return 4*np.arctan(np.exp(b))

f_1 = lambda a, t : analytical_solution(a,t)
f_2 = lambda b, t : analytical_solution(b,t)
u_0 = lambda x : analytical_solution(x,0)
u_1 = lambda x : -4*c*np.exp(x*(1-c**2)**(-1/2))/((1-c**2)**(1/2)*(1 + np.exp(2*x*(1-c**2)**(-1/2))))

def num_solution(x, t, method):
    """Numeric solution using Runge-Kutta or Runge-Kutta-Nystrøm schemes, solved with 'method' step.
    
    Input parameters:
    t: time-grid
    x: x-grid
    method: callable step function for RK or RKN loop. 
    """
    assert(callable(method))
    
    N = len(t)-1
    M = len(x)-2
    k = t[1] - t[0]
    h = x[1] - x[0]
    
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    v_left = lambda t : f_1(x[0],t)/h**2
    v_right = lambda t : f_2(x[-1],t)/h**2
    g = lambda t, v : np.concatenate(([-np.sin(v[0])+v_left(t)],-np.sin(v[1:-1]),[-np.sin(v[-1])+v_right(t)]))
    
    F = None
    if (method==RK2_step) or (method==RK3_step) or (method==RK4_step):
        F = lambda t, y : np.array([y[1], Ah @ y[0] + g(t,y[0])])
    elif (method==RKN12_step) or (method==RKN34_step):
        F = lambda t, v : Ah @ v + g(t,v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))
    #Initial conditions
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    assert(f_1(x[0],0)==u_0(x[0]))
    assert(f_2(x[-1],0)==u_0(x[-1]))
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t[i], Y[i,:,:], F)
    
    Usol = Y[:,0,:]
    
    #Insert B.C
    Usol = np.insert(Usol,0,f_1(x[0],t),axis=1)
    Usol = np.column_stack((Usol,f_2(x[-1],t)))

    return Usol

#Løsningen divergerer når N<M, noe som er feil eller er det bare CFL-condition som spiller inn-evt. dispersion som er problemet.
#RK2; M/N <= 0.65
#RK3; M/N <= 1.8
#RK4; M/N <= 2.6
#RKN12; ...
#RKN34; ...
'''
N = 10
M = 20
T = 5

x = np.linspace(-5,5,M+2)
t = np.linspace(0,T,N+1)

U = num_solution(x, t, RK4_step)

plt.plot(x,U[-1,:], label = "Numerical")
plt.plot(x,analytical_solution(x,t[-1]), label = "analytical", linestyle = "dashed")
plt.legend()
plt.show()
'''

def plot_order(Ndof, error_start, order, label, color):
    """Plots Ndof^{-order} from a starting point."""
    const = (error_start)**(1/order)*Ndof[0]
    plt.plot(Ndof, (const*1/Ndof)**order, label=label, color=color, linestyle='dashed')

def h_refinement(method):
    """Preforms h-refinement and plots the result."""
    M = np.array([8,16,32,64,128])
    err = np.zeros(len(M))
    T = 5
    N = 1000
    t = np.linspace(0, T, N + 1)
    for i,m in enumerate(M):
        x = np.linspace(-5,5,m+2) 
        U = num_solution(x,t,method)
        u = analytical_solution(x,t[-1])
        err[i] = la.norm(U[-1,:]-u)/la.norm(u)

    plt.plot(M, err, marker = 'o', label = "$||u - U||_{L_2}$")
    plot_order(M, err[0], 2, "$O(M^{-2})$", "red")
    plt.xlabel("$M$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

#h_refinement(RK2_step)

def t_refinement(method,bound): #t-refinement doesn't work yet
    """Preforms t-refinement and plots the result."""
    N = np.array([128,256,500,600,700])
    err = np.zeros(len(N))
    T = 5
    M = N*bound
    M = M.astype(int)
    
    for i,n in enumerate(N):
        x = np.linspace(-5,5,M[i]+2)
        t = np.linspace(0, T, n + 1)
        U = num_solution(x,t,method)
        u = analytical_solution(x,t[-1])
        err[i] = la.norm(U[-1,:]-u)/la.norm(u)

    plt.plot(N*M, err, marker = 'o', label = "$||u - U||_{L_2}$")
    plot_order(N*M, err[0], 1, "$O(M^{-2})$", "red")
    plt.xlabel("$N$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

#t_refinement(RK2_step,0.65)