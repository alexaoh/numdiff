"""Implementation of RK2, RK3 and RK4. Task 2c) in part 2 of semester project."""

import numpy as np

def RK2_step(t, y, h, f):
    """One step in RK2-method."""
    k1 = f(t, y)
    k2 = f(t + h, y + h*k1)
    return y + (h/2)*(k1+k2)

def RK3_step(t, y, h, f):
    """One step in RK3-method."""
    k1 = f(t, y)
    k2 = f(t + (h/2), y + (h/2)*k1)
    k3 = f(t + h, y - k1 + 2*k2)
    return y + (h/6)*(k1 + 4*k2 + k3)

def RK4_step(t, y, h, f):
    """One step in RK4-method.""" 
    k1 = f(t, y)
    k2 = f(t + (h/2), y + (h/2)*k1)  
    k3 = f(t + (h/2), y + (h/2)*k2) 
    k4 = f(t + h, y + h*k3) 
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def runge_kutta(f,t0,y0,h,nsteps, method = RK4_step):
    """Runge Kutta solution, solved with 'method' step. Standard step function is RK4.
    
    Input parameters:
    f: callable function.
    t0: initial time.
    y0: initial y.
    h: step length.
    nsteps: number of steps.
    method: callable step function for Runge Kutta loop. 
    """
    assert(callable(method))
    assert(callable(f))

    m = len(y0)
    Y = np.zeros((nsteps+1,m))
    T = np.zeros(nsteps+1)
    T[0] = t0
    Y[0] = y0
    for i in range(nsteps):
        Y[i+1] = method(T[i], Y[i], h, f)
        T[i+1] = T[i] + h
    return T,Y
