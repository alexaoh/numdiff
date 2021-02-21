"""Numerical solution using the given difference method in task 1a."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def f(x):
    """Right hand side of 1D Poisson equation."""
    return np.cos(2*np.pi*x) + x


def anal_solution(x, alpha = 0, sigma = 0):
    """Analytical solution of the Possion equation with given Neumann BC."""
    return -1/(4*np.pi**2)*np.cos(2*np.pi*x) + 1/6*x**3 + (sigma - 1/2)*x + alpha + 1/(4*np.pi**2)


def num_solution(x, M, alpha, sigma):
    """Numerical solution of the Possion equation with given Neumann BC."""

    h = 1/(M+1)

    # Construct Ah. 
    data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M+1, M+1).toarray()*1/h**2
    Ah[-1, -3] = -h/2
    Ah[-1, -2] = 2*h
    Ah[-1, -1] = -3*h/2

    # Construct f.
    f_vec = np.full(M+1, f(x[1:]))
    f_vec[0] = f_vec[0] - alpha/h**2
    f_vec[-1] = sigma

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    return Usol

M = 40
x = np.linspace(0, 1, M+2) # Make 1D grid.

alpha = 0
sigma = 0

def check(save = False):
    """Encapsulate the plotting of the solution under development for ease of use."""
    plt.plot(x, anal_solution(x, alpha, sigma), label="An", color = "black", marker = "o", linewidth = 2)
    plt.plot(x, num_solution(x, M, alpha, sigma), label="Num", color = "red", linestyle = "dotted", marker = "o", linewidth = 3)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend()
    if save:
        plt.savefig("solutionsTask1a.pdf")
    plt.show()
    # Looks good!

#### Make convergence plots for both norms. 
def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2) 
    # this is really just the regular Frobenius norm.
    # DELETE this function if we are allowed to use the Frobenius norm implemented in numpy. 
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def e_l(U, u):
    """Relative error e_l.
    
    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    #disc_l2_norm(u-U)/disc_l2_norm(u)
    # The line above is kept in case we are not allowed to use the Frobenius norm in numpy. 
    return la.norm(u-U)/la.norm(u)

def cont_L2_norm(v, left, right):
    """Continuous L2 norm of v(x) between left and right. """
    integrand = lambda x: v(x)**2
    return np.sqrt(quad(integrand, left, right)[0])

def e_L(U, u, left, right):
    """Relative error e_L.
    
    U: Approximate numerical solution.
    u: Function returning analytical solution. 
    """
    f = lambda x : u(x, alpha, sigma) - U(x)
    numer = cont_L2_norm(f, left, right)
    denom = cont_L2_norm(u, left, right)

    return numer/denom

M = np.arange(2, 1012/5, 10, dtype = int)
discrete_error = np.zeros(len(M))
cont_error = np.zeros(len(M))

for i, m in enumerate(M):
    x = np.linspace(0, 1, m+2)
    Usol = num_solution(x, M = m, alpha = alpha, sigma = sigma)
    analsol = anal_solution(x, alpha, sigma)

    discrete_error[i] = e_l(Usol, analsol)

    interpU = interp1d(x, Usol, kind = 'cubic')
    cont_error[i] = e_L(interpU, anal_solution, x[0], x[-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(M, discrete_error, label=r"$e^r_l$", color = "blue", marker = "o", linewidth = 3)
ax.plot(M, cont_error, label = r"$e^r_{L_2}$", color = "yellow", linestyle = "--", marker = "o", linewidth = 2)
ax.plot(M, (lambda x: 1/x**2)(M), label=r"$O$($h^2$)", color = "red", linewidth = 2)
ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
ax.set_xlabel("Number of points M")
plt.legend()
plt.grid() 
plt.savefig("loglogtask1a.pdf")
plt.show() 
