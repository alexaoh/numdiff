import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

def initial(x):
    return np.exp(-400*(x - 1/2)**2)


def RK4_step(v, k, f, t_i, M, h):
    '''Used as method in numerical_solution''' 
    s1 = f(t_i, v)
    s2 = f(t_i + k/2, v + (k / 2) * s1)  
    s3 = f(t_i + k/2, v + (k / 2) * s2) 
    s4 = f(t_i + k, v + k * s3) 
    return v + (k / 6) * (s1 + (2 * s2) + (2 * s3) + s4)

def numSol(F, v0, tGrid, h, M, method): 
    '''Solves the ODE \dot{v} = f(t,v) with a specified method'''
    N = len(tGrid)
    k = tGrid[1] - tGrid[0]
    vList = np.zeros((N,M))
    vList[0,:] = v0
    
    for i in range(N-1):
        val = method(vList[i,:], k, F, tGrid[i], M, h)  

        vList[i+1,:] = val       

    return vList

def F(t_i, v):
    result = np.zeros(M)
    result[0] = - v[1]*v[2]
    result[-1] = v[-1]*v[-2]

    for i in range(1, M - 1):
        result[i] = (-v[i] * (v[i + 1] - v[i - 1]))
    
    return 1/(2*h) * result


def plotTail(n, interval, sol, tGrid, xGrid):
    '''Plots the n last iterations of the solution'''
    for i in range(1, n*interval, interval):
        plt.plot(xGrid, sol[-i, :], label = f"$t = {tGrid[-i]}$")
    
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')
    plt.legend()
    #plt.show()


M = 1000
N = 1000
xGrid = np.linspace(0, 1, M + 2)
h = xGrid[1] - xGrid[0]
tGrid = np.linspace(0,0.060, N)

v0 = np.array([initial(x) for x in xGrid[1:-1]])


sol = numSol(F, v0, tGrid, h, M, RK4_step)

# Adding endpoints
zeros = np.zeros((N,1))
sol = np.hstack((zeros, sol))
sol = np.hstack((sol, zeros))


plotTail(4, 50, sol, tGrid, xGrid)



# Solve with scipy:
# Looks about right.

'''
t_bound = tGrid[-1]

scipySol = RK45(F, tGrid[0], v0, t_bound)

while scipySol.status != "finished":
    scipySol.step()

y = scipySol.y
zeros = np.zeros(1)
y = np.hstack((zeros, y))
y = np.hstack((y, zeros))
plt.plot(xGrid, y, label = f"R45 t = {t_bound}", linestyle = "dotted")
print(scipySol.t)
print(scipySol.status)

plt.legend()
'''

plt.show()

