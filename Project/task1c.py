"""Modify code from a) with different boundary conditions.

In this case we have Neumann on both sides, which corresponds to 
the simple case example in 3.1.2 in BO's note.
"""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utils import SOR, forward_subs, backward_subs, mylu
from scipy.sparse.linalg import cg

def f(x):
    """Right hand side of 1D Poisson equation."""
    return np.cos(2*np.pi*x) + x


def anal_solution(x):
    """Analytical solution of the Possion equation with given Neumann BC."""
    return -1/(4*np.pi**2)*np.cos(2*np.pi*x) + 1/6*x**3 # Utelate den siste konstanten som kan være hva som helst. 


def num_solution(x, M):
    """Numerical solution of the Possion equation with given Neumann BC."""

    h = 1/(M+1)

    # Construct Ah. 
    data = np.array([np.full(M+2, 1), np.full(M+2, -2), np.full(M+2, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M+2, M+2).toarray()*1/h**2
    Ah[0,0] = Ah[-1,-1] = -h
    Ah[0,1] = Ah[-1,-2] = h
    
    print(Ah)
    # Construct f.
    f_vec = np.full(M+2, f(x))
    f_vec[0] = (h/2)*f_vec[0]      #B.C sigma_0 = 0
    f_vec[-1] = (h/2)*f_vec[-1] - 1/2 #BC sigma_1 = 1/2
    
    trapezoidal_integral = sum(f(x[1:-1]))*h + (h/2)*(f(x[0])+f(x[-1])) 
    print(trapezoidal_integral) #it's approx 0.5 as it should
    
    # Solve linear system. 
    #Usol = la.solve(Ah, f_vec) #Ah is a singular matrix. Can it be solved in another manner?

    # Trying to solve it with an iterative method, e.g. SOR.
    # This gives a solution (with high error) and it does not makes sense in the right endpoint. 
    # We need some more information about the solution to avoid it from growing this large!!
    #Usol = SOR(Ah, f_vec, 1.2, np.zeros_like(f_vec), 1, 1300)

    # Trying to solve it with a direct method other than la from Numpy. 
    # The direct method does obviously not work, as seen. 
    # LU, P = mylu(Ah)
    # c = forward_subs(LU, P, f_vec)
    # Usol = backward_subs(LU, P, c)
    # print(Usol)

    Usol = cg(Ah, f_vec)[0]
    print(Usol)
    return Usol

M = 150
x = np.linspace(0, 1, M+2) # Make 1D grid.

num_solution(x, M)

def check():
    """Encapsulate the plotting under development for ease of use."""
    plt.plot(x, num_solution(x, M), label="Num", color = "red")
    plt.plot(x, anal_solution(x), label="An", color = "black", linestyle = "dotted")
    plt.legend()
    plt.show()

check()  #Gives an error; numpy.linalg.LinAlgError: Singular Matrix
'''
#### Make convergence plots for both norms. 
def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2) 
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def e_l(U, u):
    """Relative error e_l.
    
    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    # Could perhaps just use the regular Frobenius norm for this also?
    #la.norm(u-U)/la.norm(u)
    # Below it is seen that the regular Frobenius norm and the function disc_l2_norm give the same results (when plotted.)
    # This should be deleted later, but was to test whether or not they give the same results. 
    # Decide which to use! (Perhaps ask if we are allowed to use functions from numpy in general). 
    return disc_l2_norm(u-U)/disc_l2_norm(u), la.norm(u-U)/la.norm(u)

M = np.arange(2, 1012, 10, dtype = int)
discrete_error = np.zeros(len(M))
discrete_errorF = np.zeros(len(M))
for i, m in enumerate(M):
    x = np.linspace(0, 1, m+2)
    Usol = num_solution(x, M = m)
    analsol = anal_solution(x)
    discrete_error[i], discrete_errorF[i] = e_l(Usol, analsol)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(M, discrete_error, label="el", color = "red")
ax.plot(M, discrete_errorF, label="F", color = "black", linestyle = "dotted")
plt.legend()
plt.grid() # Looks like this is of convergence order 1. 
# I expected it to be of order 2, since we have used a central difference. Not sure why it does not get this order. 
plt.show() 
'''