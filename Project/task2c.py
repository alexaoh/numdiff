import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
#from integrators import RK4_step

def initial(x):
    return np.exp(-400*(x - 1/2)**2)

def RK4_step(k, t_i, y, f):  #<--- This function can be imported from integrators.py after we merge the files from Part 2
    '''Used as method in numerical_solution''' 
    s1 = f(t_i, y)
    s2 = f(t_i + k/2, y + (k / 2) * s1)  
    s3 = f(t_i + k/2, y + (k / 2) * s2) 
    s4 = f(t_i + k, y + k * s3) 
    return y + (k / 6) * (s1 + (2 * s2) + (2 * s3) + s4)

def numSol(x, t, method):   #Obs, initial conditions does not fit boundary conditions perfectly
    '''Solves the ODE \dot{v} = f(t,v) with a specified method'''
    M = len(x)-2
    N = len(t)-1
    h = x[1] - x[0]
    k = t[1] - t[0]
    
    vList = np.zeros((N+1,M+2))
    
    #Solves for internal grid points
    vList[0,1:-1] = initial(x[1:-1])
    F = lambda t, v : np.concatenate(([-v[0]*v[1]], -v[1:-1]*(v[2:] - v[0:-2]), [v[-1]*v[-2]]))/(2*h)
    
    for i in range(N):
        vList[i+1,1:-1] = method(k, t[i], vList[i,1:-1], F)  
    
    #Add B.C
    zeros = np.zeros_like(t)
    vList[:,0] = vList[:,-1] = zeros
    return vList

def plotTail(n, interval, sol, xGrid, tGrid):
    '''Plots the n last iterations of the solution'''
    for i in range(1, n*interval, interval):
        plt.plot(xGrid, sol[-i, :], label = f"$t = {tGrid[-i]}$")
    
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')
    plt.legend()
    plt.show()

# Plot breaking point of solution
M = 1000
N = 1000
T = 0.06
x = np.linspace(0, 1, M+2)
t = np.linspace(0, T, N+1)

sol = numSol(x, t, RK4_step)

plotTail(4, 50, sol, x, t)



# Solve with scipy RK45
def scipy_solution(x,t0,t_bound):
    """Solves the ODE with scipy's integrated function RK45 and plot the solution at t_bound"""
    
    h = x[1] - x[0]
    F = lambda t, v : np.concatenate(([-v[0]*v[1]], -v[1:-1]*(v[2:] - v[0:-2]), [v[-1]*v[-2]]))/(2*h)
    
    scipySol = RK45(F, t0, initial(x[1:-1]), t_bound)

    while scipySol.status != "finished":
        scipySol.step()

    y = scipySol.y
    zeros = np.zeros(1)
    y = np.hstack((zeros, y))
    y = np.hstack((y, zeros))
    plt.plot(x, y, label = f"R45 t = {t_bound}", linestyle = "dotted")
    print(scipySol.t)
    print(scipySol.status)

    plt.legend()
    plt.show()
    
#scipy_solution(x, 0, t[-1]) #Looks about right




