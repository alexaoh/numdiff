"""Modify code from a) with different boundary conditions.

In this case we have Dirichlet on both sides, which corresponds to 
the simple case example in 3.1 in BO's note.
"""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def f(x):
    """Right hand side of 1D Poisson equation."""
    return np.cos(2*np.pi*x) + x


def anal_solution(x):
    """Analytical solution of the Possion equation with given Neumann BC."""
    return -1/(4*np.pi**2)*np.cos(2*np.pi*x) + 1/6*x**3 + (-1/6)*x + (1 + 1/(4*np.pi**2))


def num_solution(x, M):
    """Numerical solution of the Possion equation with given Neumann BC."""

    h = 1/(M+1)

    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2

    # Construct f.
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - 1/h**2
    f_vec[-1] = f_vec[-1] - 1/h**2

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, 1)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol, 1)

    return Usol

M = 1000
x = np.linspace(0, 1, M+2) # Make 1D grid.

num_solution(x, M)

def check():
    """Encapsulate the plotting under development for ease of use."""
    plt.plot(x, num_solution(x, M), label="Num", color = "red")
    plt.plot(x, anal_solution(x), label="An", color = "black", linestyle = "dotted")
    plt.legend()
    plt.show()
    # Looks good!

#check()

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

def cont_L2_norm(v, left, right):
    """Continuous L2 norm of v(x) between left and right. """
    integrand = lambda x: v(x)**2
    return np.sqrt(quad(integrand, left, right)[0])

def e_L(U, u, left, right):
    """Relative error e_L.
    
    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    f = lambda x : u(x) - U(x)
    numer = cont_L2_norm(f, left, right)
    denom = cont_L2_norm(u, left, right)

    return numer/denom


M = np.arange(2, 1012, 10, dtype = int)
discrete_error = np.zeros(len(M))
discrete_errorF = np.zeros(len(M))
cont_error = np.zeros(len(M))

for i, m in enumerate(M):
    x = np.linspace(0, 1, m+2)
    Usol = num_solution(x, M = m)
    analsol = anal_solution(x)
    discrete_error[i], discrete_errorF[i] = e_l(Usol, analsol)

    interpU = interp1d(x, Usol, kind = 'cubic')
    cont_error[i] = e_L(interpU, anal_solution, x[0], x[-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(M, discrete_errorF, label="e_l", color = "blue")
ax.plot(M, cont_error, label = "e_L2", color = "red", linestyle = "dotted")
plt.legend()
plt.grid() # Looks like this is of convergence order 1. 
# I expected it to be of order 2, since we have used a central difference. Not sure why it does not get this order. 
plt.savefig("loglogtask1b.pdf")
plt.show() 
