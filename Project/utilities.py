"""Functions used throughout."""
import numpy as np
import numpy.linalg as la
from scipy.integrate import quad

def cont_L2_norm(v, left, right):
    """Continuous L2 norm of v(x) between left and right. """
    integrand = lambda x: v(x)**2
    return np.sqrt(quad(integrand, left, right)[0])

def e_L(U, u, left, right):
    """Relative error e_L.
    
    U: Approximate numerical solution.
    u: Function returning analytical solution. 
    """
    f = lambda x : u(x) - U(x)
    numer = cont_L2_norm(f, left, right)
    denom = cont_L2_norm(u, left, right)

    return numer/denom

def e_l(U, u):
    """Relative error e_l.
    
    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    return la.norm(u-U)/la.norm(u)
