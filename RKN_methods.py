"""Implementation of RKN methods. Task 2e) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import spdiags

c = 0.5
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the plus-sign as solution."""
    b = (x-c*t)*(1-c**2)**(-1/2) 
    return 4*np.arctan(np.exp(b))

f_1 = lambda a, t : analytical_solution(a,t)
f_2 = lambda b, t : analytical_solution(b,t)
u_0 = lambda x : analytical_solution(x,0)
u_1 = lambda x : -4*c*np.exp(x*(1-c**2)**(-1/2))/((1-c**2)**(1/2)*(1 + np.exp(2*x*(1-c**2)**(-1/2))))

def RKN12_step(k, t_i, y, f):
    s1 = f(t_i, y[0])
    
    y_der_new = y[1] + k*s1
    y_new = y[0] + k*y[1] + k**2*(s1/2)
    return np.array([y_new, y_der_new])

delta = (1/12)*(2 - 4**(1/3)-16**(1/3))
def RKN34_step(k, t_i, y, f):
    s1 = f(t_i+(1/2-delta)*k, y[0] + (1/2-delta)*k*y[1])
    s2 = f(t_i + (1/2)*k, y[0] + (1/2)*k*y[1]+k**2/(24*delta)*s1)
    s3 = f(t_i + (1/2 + delta)*k, y[0] + (1/2+delta)*k*y[1] + k**2*(1/(12*delta)*s1 + (delta-1/(12*delta))*s2))
    
    y_der_new = y[1] + k*((s1+s3)/(24*delta**2) + (1-1/(12*delta**2))*s2)
    y_new = y[0] + k*y[1] + k**2*((1-1/2+delta)/(24*delta**2)*s1 + ((1-1/(12*delta**2))*(1-1/2))*s2 + (1-1/2-delta)/(24*delta**2)*s3)
    return np.array([y_new, y_der_new])
    
def RKN_solver(x, t, method):
    """Runge Kutta solution, solved with 'method' step. Standard step function is RK4.
    
    Input parameters:
    x: x-grid
    t: time-grid
    f: callable function
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
    #assert(callable(f))
    
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
    F = lambda t, v : Ah @ v + g(t,v)
        
    Y = np.zeros((N+1,2,M))
    #Initial conditions
    Y[0,0,:] = u_0(x[1:-1]) 
    Y[0,1,:] = u_1(x[1:-1]) 
    
    assert(f_1(x[0],0)==u_0(x[0]))
    assert(f_2(x[-1],0)==u_0(x[-1]))
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t[i], Y[i,:,:], F)

    Usol = Y[:,0,:]
    
    #Insert B.C
    Usol = np.insert(Usol,0,f_1(x[0],t),axis=1)
    Usol = np.column_stack((Usol,f_2(x[-1],t)))
    
    return Usol

#Noe rart skjer når N<<M
M = 200
N = 200
T = 5
x = np.linspace(-2,2,M+2)
t = np.linspace(0,T,N+1)

U = RKN_solver(x,t,RKN12_step)
plt.plot(x,U[-1,:])
plt.show()

'''
M = np.array([8,16,32,64,128])
err = np.zeros(len(M))
for i,m in enumerate(M):
    x = np.linspace(-2,2,m+2)
    U = RKN_solver(x,t,F,RKN12_step)
    u = analytical_solution(x,t[-1])
    err[i] = la.norm(U[-1,:]-u)/la.norm(u)

plt.plot(M, err)
plt.plot(M,(1/M)**(2),linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''