"""Task 5."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.sparse.linalg import spsolve 
from scipy.linalg import toeplitz, solve_toeplitz
import numpy as np
import numpy.linalg as la
from scipy.interpolate import interp1d 
from scipy.integrate import quad
from scipy.integrate import quadrature
import matplotlib.pyplot as plt

def plot_order(Ndof, error_start, order, label, color):
    const = (error_start)**(1/order)*Ndof[0]
    plt.plot(Ndof, (const*1/Ndof)**order, label=label, color=color, linestyle='dashed')

def cont_L2_norm(v, left, right):
    """Continuous L2 norm of v(x) between left and right. """
    integrand = lambda x: v(x)**2
    return np.sqrt(quad(integrand, left, right)[0])

def e_L2(U, u, left, right):
    """L2 error on the interval [left, right].
    
    U: Approximate numerical solution.
    u: Function returning analytical solution. 
    """
    f = lambda x : u(x) - U(x)
    err = cont_L2_norm(f, left, right)
    return err

def FEM_solver_equidistant(BC, f, x):
    '''FEM solver which only works with equidistant grid x.'''
    # Construction below assumes equidistant grid
    N = len(x)
    h = 1/(N-1)
    
    data = np.array([np.full(N - 2, -1), np.full(N - 2, 2), np.full(N -2, -1)])
    diags = np.array([-1, 0, 1])
    A = 1/h * spdiags(data, diags, N - 2, N - 2).toarray()
    
    # Construct rhs

    rhs = np.zeros(N - 2)
    for i in range(N - 2):
        rhs[i] = h * f(x[i + 1])
    
    rhs[0] += 1/h * BC[0]
    rhs[-1] += 1/h * BC[1]

    u = la.solve(A, rhs)
    u = np.hstack((np.array(BC[0]), u))
    u = np.hstack((u, np.array(BC[1])))

    return u




def FEM_solver_Dirichlet(BC, f, x):
    '''General FEM solver with Dirichlet BC.'''
    N = len(x)
    # Construct A
    A = np.zeros((N,N))
    for i in range(N - 1):
        A[i:(i + 2), i:(i + 2)] +=  1/(x[i+1] - x[i]) * np.array([[1, -1], [-1, 1]])

    rhs = np.zeros(N)
    for i in range(N - 1):
        # Functions to do Gaussian quadrature on:
        v1 = lambda y: (x[i + 1] - y) * f(y)
        v2 = lambda z: (z -x[i]) * f(z)
        rhs[i:(i+2)] += 1/(x[i + 1] - x[i]) * np.array([quadrature(v1, x[i], x[i + 1])[0], quadrature(v2, x[i], x[i + 1])[0]])

    #This could prabably be coded more efficiently
    BC_vec = np.zeros(N)
    BC_vec[0] = BC[0]
    BC_vec[-1] = BC[1]

    rhs = rhs - A @ BC_vec

    # Implementing BC:
    A = A[1:-1,1:-1]
    rhs = rhs[1:-1]

    # Solve system
    u = la.solve(A, rhs)
    u = np.hstack((np.array(BC[0]), u))
    u = np.hstack((u, np.array(BC[1])))

    return u



# b)

def UFEM(N_list, BC, f, anal_sol, x_interval):
    '''Conducts FEM with uniform refinement in 1D.'''
    err_list = []

    for N in N_list:
        x = np.linspace(x_interval[0], x_interval[1], N)
        U = FEM_solver_Dirichlet(BC, f, x)
        U_interp = interp1d(x, U, kind = 'linear')
        '''
        plt.plot(x, U_interp(x), label = "num", marker = 'o')
        plt.plot(x, anal_sol(x), label = "anal", linestyle = "dotted")
        plt.legend()
        plt.show()
        '''
        err_list.append(e_L2(U_interp, anal_sol, x[0], x[-1]))

    plt.plot(N_list, err_list, marker = 'o', label = "$||u - u_h||_{L_2}$")
    plot_order(np.array(N_list), err_list[0], 2, "$O(h^{-2})$", color = "red")

    plt.title("UFEM")
    plt.xlabel("$N$")
    #plt.ylabel("$||u - u_h||_{L_2}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

# AFEM:
def calc_cell_errors(U, u, x):
    '''Calulculates the error for each cell by taking the L_2 norm.'''
    n = len(x) - 1 # Number of cells
    cell_errors = np.zeros(n)

    for i in range(n):
        cell_errors[i] = e_L2(U, u, x[i], x[i + 1])

    return cell_errors

def AFEM(N0, steps, alpha, type, f, anal_sol, x_interval):
    '''Conducts FEM with adaptive refinement in 1D steps times.'''
    err_list = []
    N_list = []
    x = np.linspace(x_interval[0], x_interval[1], N0)

    for i in range(steps):
        U = FEM_solver_Dirichlet(BC, f, x)
        U_interp = interp1d(x, U, kind = 'linear')

        '''
        plt.plot(x, U_interp(x), label = "num", marker = 'o')
        plt.plot(x, anal_sol(x), label = "anal", linestyle = "dotted")
        plt.legend()
        plt.show()
        '''
        cell_errors = calc_cell_errors(U_interp, anal_sol, x)
        err = e_L2(U_interp, anal_sol, x[0], x[-1])

        N_list.append(len(x))
        err_list.append(err)

        if type == 'avg':
            err = 1/N_list[i] * err
        elif type == 'max':
            err = max(cell_errors)
        else:
            raise Exception("Invalid type.")
        
        x = list(x)
        k = 0 # index for x in case we insert points.
        for j in range(len(cell_errors)):
            if cell_errors[j] > alpha * err:
                x.insert(k+1, x[k] + 0.5 * (x[k+1] - x[k]))
                k += 1
            k += 1
        x = np.array(x)

        '''
        plt.bar([i for i in range(len(cell_errors))], cell_errors)
        plt.scatter(x, np.zeros(len(x)), s = 1)
        plt.show()
        '''
    plt.plot(N_list, err_list, marker = 'o', label = "$||u - u_h||_{L_2}$")
    plot_order(np.array(N_list), err_list[0], 2, "$O(h^{-2})$", color = "red")

    plt.title(str(steps) + '-step AFEM with ' + type + '-error and ' + r'$\alpha =$' + str(alpha))
    plt.xlabel("$N$")
    #plt.ylabel("$e_{L_2}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

## b)
# UFEM
anal_sol = lambda x: x**2
f = lambda x: -2
x_interval = [0,1]
BC = [0, 1]
N_list = [2**i for i in range(3, 12)]
#UFEM(N_list, BC, f, anal_sol, x_interval)

# Average AFEM
steps = 7
alpha = 1
N0 = 20
#AFEM(N0, steps, alpha, 'avg', f, anal_sol, x_interval)

# Maximum AFEM
alpha = 0.7
#AFEM(N0, steps, alpha, 'max', f, anal_sol, x_interval)


## c)
#UFEM
anal_sol = lambda x: np.exp(-100 * x**2)
f = lambda x: - (40000*x**2 - 200) * np.exp(-100 * x**2)
x_interval = [-1, 1]
BC = [np.exp(-100), np.exp(-100)]
N_list = [2**i for i in range(3, 12)]
#UFEM(N_list, BC, f, anal_sol, x_interval)

# Average AFEM
steps = 7
alpha = 1
N0 = 20
#AFEM(N0, steps, alpha, 'avg', f, anal_sol, x_interval)

#Maximum AFEM
alpha = 0.7
#AFEM(N0, steps, alpha, 'max', f, anal_sol, x_interval)

## d)
#UFEM
anal_sol = lambda x: np.exp(-1000 * x**2)
f = lambda x: - (4000000 * x**2 - 2000) * np.exp(-1000 * x**2)
x_interval = [-1, 1]
BC = [np.exp(-1000), np.exp(-1000)]
N_list = [2**i for i in range(3, 12)]
#UFEM(N_list, BC, f, anal_sol, x_interval)

# Average AFEM
steps = 7
alpha = 1
N0 = 20
#AFEM(N0, steps, alpha, 'avg', f, anal_sol, x_interval)
# Maximum AFEM
alpha = 0.7
#AFEM(N0, steps, alpha, 'max', f, anal_sol, x_interval)


## e)
# UFEM
anal_sol = lambda x: x**(2/3)
f = lambda x: - 2/9 * x**(-4/3)
x_interval = [0, 1]
BC = [0, 1]
N_list = [2**i for i in range(3, 12)]
UFEM(N_list, BC, f, anal_sol, x_interval)

# Average AFEM
steps = 7
alpha = 1
N0 = 20
#AFEM(N0, steps, alpha, 'avg', f, anal_sol, x_interval)

# Maximum AFEM
alpha = 0.7
#AFEM(N0, steps, alpha, 'max', f, anal_sol, x_interval)






