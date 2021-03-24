"""Task 5."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.sparse.linalg import spsolve 
from scipy.linalg import toeplitz, solve_toeplitz
import numpy as np
import numpy.linalg as la
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import matplotlib.pyplot as plt

def plot_order(Ndof, error_start, order, label, color):
    const = (error_start)**(1/order)*Ndof[0]
    plt.plot(Ndof, (const*1/Ndof)**order, label=label, color=color, linestyle='dashed')

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

N_list = [2**i for i in range(3, 12)]



def FEM_solver_Dirichlet(BC, f, x):

    N = len(x)
    h = 1/N
    # Construct A
    data = np.array([np.full(N - 2, -1), np.full(N - 2, 2), np.full(N -2, -1)])
    diags = np.array([-1, 0, 1])
    A = 1/h * spdiags(data, diags, N - 2, N - 2).toarray()
    # Construct rhs
    rhs = np.zeros(N - 2)
    for i in range(N - 2):
        rhs[i] = h * f(x[i + 1])
    rhs[0] += 1/h * BC[0]
    rhs[-1] += 1/h * BC[1]


    # Solve system
    u = la.solve(A, rhs)
    u = np.hstack((np.array(BC[0]), u))
    u = np.hstack((u, np.array(BC[1])))

    #plt.plot(x,u)
    #plt.plot(x,anal_sol(x))
    #plt.show()


    return u



# b)

def UFEM():
    '''Conducts FEM with uniform refinement in 1D.'''
    anal_sol = lambda x: x**2
    f = lambda x: -2
    BC = [0, 1]
    N_list = [2**i for i in range(3, 12)]
    err_list = []

    for N in N_list:
        x = np.linspace(0, 1, N)
        u = FEM_solver_Dirichlet(BC, f, x)
        u_interp = interp1d(x, u, kind = 'cubic')

        err_list.append(e_L(u_interp, anal_sol, x[0], x[-1]))

    plt.plot(N_list, err_list, marker = 'o')
    plot_order(np.array(N_list), err_list[0], 1, "$O(h^{-2})$", color = "red")

    plt.xlabel("$N$")
    plt.ylabel("$e_{L_2}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

# AFEM:
def calc_cell_errors(u, U, x):
    '''Calulculates the error for each cell by taking the L_2 norm.'''
    n = len(x) - 1 # Number of cells
    cell_errors = np.zeros(n)

    for i in range(n):
        cell_errors[i] = e_L(U, u, x[i], x[i +1])

    return cell_errors

def AFEM(N0, steps, alpha, type):
    '''Conducts FEM with adaptive refinement in 1D steps times.'''
    anal_sol = lambda x: x**2
    f = lambda x: -2
    BC = [0, 1]
    N_list = []
    err_list = []
    x = np.linspace(0, 1, N0)

    for i in range(steps):
        U = FEM_solver_Dirichlet(BC, f, x)
        U_interp = interp1d(x, U, kind = 'cubic')
        cell_errors = calc_cell_errors(anal_sol, U_interp, x)
        err = e_L(U_interp, anal_sol, x[0], x[-1])

        N_list.append(len(x))
        err_list.append(err)

        if type == 'avg':
            err = 1/N_list[i] * err
        elif type == 'max':
            err = np.max(cell_errors)
        else:
            raise Exception("Invalid type.")
        
        x = list(x)
        for j in range(len(cell_errors)):
            if cell_errors[i] > alpha * err:
                x.insert(j+1, x[j] + 0.5 * (x[j+1] - x[j]))
        x = np.array(x)
    

    plt.plot(N_list, err_list, marker = 'o')
    plot_order(np.array(N_list), err_list[0], 1, "$O(h^{-2})$", color = "red")

    plt.xlabel("$N$")
    plt.ylabel("$e_{L_2}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


UFEM()
#AFEM(20, 10, 1, 'avg')




