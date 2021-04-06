"""Implementation of RKN methods. Task 2e) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

c = 0.5
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the plus-sign as solution."""
    b = (x-c*t)*(1-c**2)**(-1/2) 
    return 4*np.arctan(np.exp(b))

f_1 = lambda a, t : analytical_solution(a,t)
f_2 = lambda b, t : analytical_solution(b,t)
u_0 = lambda x : analytical_solution(x,0)
u_1 = lambda x : -4*c*np.exp(x*(1-c**2)**(-1/2))/((1-c**2)**(1/2)*(1 + np.exp(2*x*(1-c**2)**(-1/2))))

def RKN12_step(h, k, t_i, y, y_der, f):
    s1 = f(h, t_i, y)
    y_der_new = y_der + k*s1
    y_new = y + k*y_der + h**2*(1/2)*s1
    return y_new, y_der_new

def RKN_solver(x, t, f, method):
    """Runge Kutta solution, solved with 'method' step. Standard step function is RK4.
    
    Input parameters:
    t: time-grid
    x: x-grid
    f: callable function
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
    assert(callable(f))
    
    N = len(t)-1
    k = t[1] - t[0]
    h = x[1] - x[0]
    Y = np.zeros((N+1,2,len(x)))
    
    #Initial conditions
    Y[0, 0, :] = u_0(x) #y0
    Y[0, 1, :] = u_1(x) #y_der_0
    for i in range(N):
        Y[i+1,0,1:-1], Y[i+1,1,1:-1] = method(h, k, t[i], Y[i,0,1:-1], Y[i,1,1:-1], f)
    
    #Insert B.C
    Y[:, 0, 0] = f_1(x[0],t)
    Y[:, 0, -1] = f_2(x[-1],t)
    return Y[:,0,:]

def F(h,t_i,y):
    """Expression for derivatives of y-vector = [v_m,w_m] for 1 <= m <= M+1.
    Using analytic function at boundary as boundary conditions.
    """
    res = np.zeros(len(y))
    
    res[0] = (1/h**2)*(f_1(x[0],t_i)-2*y[0]+y[1]) - np.sin(y[0])
    res[1:-1] = (1/h**2)*(y[:-2]-2*y[1:-1]+y[2:]) - np.sin(y[1:-1])
    res[-1] = (1/h**2)*(y[-2]-2*y[-1]+f_2(x[-1],t_i)) - np.sin(y[-1])
    
    return res

#Still doesent work with t-refinement
M = 200
N = 400
T = 5
x = np.linspace(-2,2,M+1)
t = np.linspace(0,T,N+1)

U = RKN_solver(x,t,F,RKN12_step)

plt.plot(x,U[-1,:])
plt.show()