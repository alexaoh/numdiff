import numpy as np
import matplotlib.pyplot as plt

def initial(x):
    return np.exp(-400*(x - 1/2)**2)


def RK4_step(v, k, f, t_i, M, h):
    '''Used as method in numerical_solution''' 
    s1 = f(v, t_i, M, h)
    s2 = f(v + (k / 2) * s1, t_i + k/2, M, h)  
    s3 = f(v + (k / 2) * s2, t_i + k/2, M, h) 
    s4 = f(v + k * s3, t_i + k, M, h) 
    return v + (k / 6) * (s1 + (2 * s2) + (2 * s3) + s4)

def numSol(F, v0, tGrid, h, M, method): 
    '''Solves the ODE \dot{v} = f(t,v) with a specified method'''
    N = len(tGrid)
    k = tGrid[1] - tGrid[0]
    vList = np.zeros((N,M))
    vList[0,:] = v0
    
    for i in range(N-1):
        print(i)
        val = method(vList[i,:], k, F, tGrid[i], M, h)  

        vList[i+1,:] = val       

    return vList

def F(v, t_i, M, h):
    result = np.zeros(M)
    result[0] = - v[1]*v[2]
    result[-1] = v[-1]*v[-2]

    for i in range(1, M - 1):
        result[i] = 1/(2*h) * (-v[i] * (v[i + 1] - v[i - 1]))
    
    return result


def plotTail(n, interval, sol, tGrid, xGrid):
    '''Plots the n last iterations of the solution'''
    for i in range(1, n*interval, interval):
        plt.plot(xGrid, sol[-i, :], label = f"$t = {tGrid[-i]}$")
    
    plt.legend()
    plt.show()


M = 1000
N = 1000
xGrid = np.linspace(0, 1, M + 2)
h = xGrid[1] - xGrid[0]
tGrid = np.linspace(0,0.065, N)

v0 = np.array([initial(x) for x in xGrid[1:-1]])
sol = numSol(F, v0, tGrid, h, M, RK4_step)

# Adding endpoints
zeros = np.zeros((N,1))
sol = np.hstack((zeros, sol))
sol = np.hstack((sol, zeros))



print(sol)
plotTail(5, 50, sol, tGrid, xGrid)