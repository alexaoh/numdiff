"""Implementation of RK2, RK3 and RK4. Task 2d) in part 2 of semester project."""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

c = 0.5  #Kan vi bestemme c selv?
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the plus-sign as solution."""
    b = (x-c*t)*(1-c**2)**(-1/2) #Valgte plus-tegnet som løsning, det er vel greit?
    return 4*np.arctan(np.exp(b))

f_1 = lambda a, t : analytical_solution(a,t)
f_2 = lambda b, t : analytical_solution(b,t)
u_0 = lambda x : analytical_solution(x,0)
u_1 = lambda x : -4*c*np.exp(x*(1-c**2)**(-1/2))/((1-c**2)**(1/2)*(1 + np.exp(2*x*(1-c**2)**(-1/2))))

def RK2_step(h, k, t_i, y, f):
    """One step in RK2-method."""
    s1 = f(h, t_i, y)
    s2 = f(h,t_i +k, y + k*s1)
    return y + (k/2)*(s1+s2)

def RK3_step(h, k, t_i,y, f):
    """One step in RK3-method."""
    s1 = f(h, t_i,y)
    s2 = f(h, t_i+k/2, y + (k/2)*s1)
    s3 = f(h, t_i+k, y - s1 + 2*s2)
    return y + (k/6)*(s1 + 4*s2 + s3)

def RK4_step(h, k, t_i, y, f):
    """One step in RK4-method.""" 
    s1 = f(h,t_i, y)
    s2 = f(h, t_i+k/2, y + (k/2)*s1)  
    s3 = f(h, t_i+k/2, y + (k/2)*s2) 
    s4 = f(h, t_i+k, y + k*s3)
    return y + (k/6)*(s1 + 2*s2 + 2*s3 + s4)

def runge_kutta(x, t, f, method = RK4_step):
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
    Y[0, 0, :] = u_0(x)
    Y[0, 1, :] = u_1(x)
    for i in range(N):
        Y[i+1,:,1:-1] = method(h, k, t[i], Y[i,:,1:-1], f) #Løser kun for de nodene innenfor kanten, setter inn B.C etterpå.
    
    #Insert B.C
    Y[:, 0, 0] = f_1(x[0],t)
    Y[:, 0, -1] = f_2(x[-1],t)
    return Y[:,0,:]

def F(h,t_i,y):
    """Expression for derivatives of y-vector = [v_m,w_m] for 1 <= m <= M+1.
    Using analytic function at boundary as boundary conditions.
    """
    res = np.zeros(np.shape(y))
    
    res[0,:] = y[1,:]
    res[1,0] = (1/h**2)*(f_1(x[0],t_i)-2*y[0,0]+y[0,1]) - np.sin(y[0,0])
    res[1,1:-1] = (1/h**2)*(y[0,:-2]-2*y[0,1:-1]+y[0,2:]) - np.sin(y[0,1:-1])
    res[1,-1] = (1/h**2)*(y[0,-2]-2*y[0,-1]+f_2(x[-1],t_i)) - np.sin(y[0,-1])
    
    return res

#Løsningen divergerer når M>N. Noe som er feil her.
#Funker derfor ikke med t-refinement
M = 200
N = 1000
T = 5
x = np.linspace(-2,2,M+1)
t = np.linspace(0,T,N+1)

U = runge_kutta(x, t, F, RK4_step)

plt.plot(x,U[-1,:])
plt.plot(x,analytical_solution(x,t[-1]))
plt.show()

'''
M = np.array([8,16,32,64,128])
err = np.zeros(len(M))
for i,m in enumerate(M):
    x = np.linspace(-2,2,m+2)
    U = runge_kutta(x,t,F,RK2_step)
    u = analytical_solution(x,t[-1])
    err[i] = la.norm(U[-1,:]-u)/la.norm(u)

plt.plot(M, err)
plt.plot(M,(1/M)**(2),linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
