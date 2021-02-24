import numpy as np
import matplotlib.pyplot as plt

def initial(x):
    return np.exp(-400*(x - 1/2)**2)


def RK4_step(v, k, f, t_i, M):
    '''Used as method in numerical_solution''' 
    s1 = f(v, t_i, M)
    s2 = f(v + (k / 2) * s1, t_i + k/2, M)  
    s3 = f(v + (k / 2) * s2, t_i + k/2, M) 
    s4 = f(v + k * s3, t_i + k, M) 
    return v + (k / 6) * (s1 + (2 * s2) + (2 * s3) + s4)

def numSol(F, v0, tGrid, h, M, method): 
    '''Solves the ODE \dot{v} = f(t,v) with a specified method'''
    N = len(tGrid)
    k = tGrid[1] - tGrid[0]
    vList = np.zeros((N,M))
    vList[0,:] = v0
    
    for i in range(N-1):
        print(i)
        val = method(vList[i,:], k, F, tGrid[i], M)  

        vList[i+1,:] = 1/(2*h) * val       

    return vList

def F(v, t_i, M):
    result = np.zeros(M)
    result[0] = - v[1]*v[2]
    result[-1] = v[-1]*v[-2]

    for i in range(1, M - 1):
        result[i] = -v[i] * (v[i + 1] - v[i - 1])
    
    return result



M = 1000
N = 10000
xGrid = np.linspace(0, 1, M + 2)
h = xGrid[1] - xGrid[0]
tGrid = np.linspace(0,0.001, N)

v0 = np.array([initial(x) for x in xGrid[1:-1]])
sol = numSol(F, v0, tGrid, h, M, RK4_step)

# Adding endpoints
zeros = np.zeros((N,1))
sol = np.hstack((zeros, sol))
sol = np.hstack((sol, zeros))

print(sol)
plt.plot(xGrid, sol[1])
plt.plot(xGrid, sol[2])
plt.plot(xGrid, sol[3])
#plt.plot(xGrid, sol[4])
#plt.plot(xGrid, sol[5])#
#plt.plot(xGrid, sol[10])
plt.show()
