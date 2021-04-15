#from crank_nicolson import * # is this needed? Delete the file and this comment if not ;)
from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import pickle # To save the reference solution.
from utilities import *

initial = (lambda x: 2*np.pi*x - np.sin(2*np.pi*x))

def theta_method_given_Q(x, t, V0, Q, theta): 
    """Solves \dot{V} = \frac{1}{h^2}QV on time axis t.

    Input: 
    x: x-axis.
    t: t-axis. 
    V0: Grid to solve on.
    Q: Matrix in right hand side of equation to solve. 
    theta: Parameter in theta-method to change the weighting of the terms. 
           Special cases: theta = 1 gives BE and theta = 1/2 gives CN. 
    """
    sol = np.zeros((len(t),len(x)))
    sol[0,:] = V0
    k = t[1]-t[0]
    h = x[1]-x[0]
    M = Q.shape[0]  #Is this really M or M+1? Not that it matters...
    r = k/h**2
    
    lhs = (np.eye(M) - theta*r*Q)
    b = (np.eye(M) + (1-theta)*r*Q)
    for n in range(len(t)-1):
        rhs = b @ sol[n, :]
        sol[n+1, :] = np.linalg.solve(lhs, rhs)
        
    return sol

def calc_sol(x, t, order, theta, plot = False):
    """ 
    order = 1: Use first oder disc. on BC.
    order = 2: Use second order disc. on BC.
    theta = 1: Backward Euler.
    theta = 1/2: Trapezoidal rule (CN).
    """
    M = len(x)-1
    # Construct Q.
    data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
    diags = np.array([-1, 0, 1])
    Q = spdiags(data, diags, M+1, M+1).toarray()
    if order == 1:
        Q[0, 0] = Q[-1, -1] = -1
        Q[0, 1] = Q[-1, -2] =  0
        Q[0, 2] = Q[-1,-3] = 1


    elif order == 2:
        Q[0, 1] = Q[-1, -2] =  2
    
    
    V0 = [initial(x) for x in x]
    sol = theta_method_given_Q(x, t, V0, Q, theta) 
  
    if plot:
        tv, xv = np.meshgrid(t,x)
        three_dim_plot(xv = xv, tv = tv, I = sol.T, label = "Numerical Solution")
    
    return sol


def save_ref_sol(Mstar, Nstar, order, theta, filename):
    """Save the reference solution to file."""
    T = 0.2
    x = np.linspace(0,1,Mstar+1)
    t = np.linspace(0,T,Nstar+1)
    ref_sol = calc_sol(x, t, order, theta) 
    with open(filename, 'wb') as fi:
        pickle.dump(ref_sol, fi)


def calc_error(M, N, filename, typ): 
    """Calculate the relative error with the reference solution.
    Input:
    M: List or scalar depending on the type of refinement.
    N: List or scalar depending on the type of refinement.
    filename: Name of the file where the reference solution has been saved. 
    type: h, t or r - refinement.
    """
    assert(typ == "t" or typ == "r" or typ == "h") # SKulle det vært noe r - refinement her?
    ref_sol = None
    Mstar = Nstar = 1000   # Change values here if you change it in the file.
    with open(filename, 'rb') as fi: 
        ref_sol = pickle.load(fi)

    if typ=='h':
        N = np.ones_like(M)*N
    elif typ=='t':
        M = np.ones_like(N)*M
    modulus = Mstar % M    # Controls that M are divisible by Mstar. This does not apply to N because we look at error in T which is similar for both sol and ref_sol
    if len(modulus[np.nonzero(modulus)]) != 0:
        print('Wrong M values.')
        return 1
        
    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))

    T = 0.2
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+1)
        t = np.linspace(0,T,N[i]+1)
        sol_1 = calc_sol(x, t, 2, 1)
        sol_2 = calc_sol(x, t, 2, 1/2)
        u_star = ref_sol[-1,0::(Mstar//M[i])]
       
        disc_err_first[i] = e_l(sol_1[-1,:], u_star)
        disc_err_second[i] = e_l(sol_2[-1,:], u_star)      
    
    MN = M*N

    plt.plot(MN, disc_err_first, label = "$e^r_l$ (BE)",color='red',marker='o')
    plt.plot(MN, disc_err_second, label = "$e^r_l$ (CN)",color='blue',marker='o')

    # These need to be changed manually!
    plot_order(MN, disc_err_first[0], 1, label = r"$\mathcal{O}(N_{dof}^{-1})$", color = 'red')
    plot_order(MN, disc_err_second[0], 2, label = r"$\mathcal{O}(N_{dof}^{-2})}$", color='blue')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('$M*N$')
    plt.ylabel(r"Error $e^r_{l}$")
    plt.legend()
    plt.grid()
    plt.show()

def compare_discr(M, N, filename):
    """Compares the first and second order discretizations of the BCs by makeing a convergence plot."""
    ref_sol = None
    with open(filename, 'rb') as fi: 
        ref_sol = pickle.load(fi)
    
    N = np.ones_like(M)*N
    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))
    T = 0.2
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+1)
        t = np.linspace(0,T,N[i]+1)
        sol_1 = calc_sol(x, t, 1, 1/2)
        sol_2 = calc_sol(x, t, 2, 1/2)
        u_star = ref_sol[-1,0::(Mstar//M[i])]
       
        disc_err_first[i] = e_l(sol_1[-1,:], u_star)
        disc_err_second[i] = e_l(sol_2[-1,:], u_star)      
    
    MN = M*N

    plt.plot(MN, disc_err_first, label = "$e^r_l$ (1st order)",color='red',marker='o')
    plt.plot(MN, disc_err_second, label = "$e^r_l$ (2nd order)",color='blue',marker='o')

    # These need to be changed manually!
    plot_order(MN, disc_err_first[0], 1, label = r"$\mathcal{O}(N_{dof}^{-1})$", color = 'red')
    plot_order(MN, disc_err_second[0], 2, label = r"$\mathcal{O}(N_{dof}^{-2})}$", color='blue')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('$M*N$')
    plt.ylabel(r"Error $e^r_{l}$")
    plt.legend()
    plt.grid()
    plt.show()


# ====| Run code below. |==== #

Mstar = Nstar = 1000
filename = 'ref_sol.pk'
# We use 2 order disc. of BC and CN for the reference solution.
#save_ref_sol(Mstar, Nstar, 2, 1/2, filename) # Only needs to be run once, or if you change Mstar

# ---| Plot solution. |--- #
x = np.linspace(0, 1, 1000)
t = np.linspace(0, 0.2, 100)
#calc_sol(x, t, 2, 1/2, True)

# ---| Compare firts and second order disc. of BCs. |--- # 
N = 1000
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
#compare_discr(M,N,filename)

# ---| h-refinement. |--- # 
N = 1000
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
#calc_error(M, N, filename, 'h')

# ---| k-refinement. |--- #
M = 1000
#N = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([4,8,16,32,64,128,256])  #does not have to be divisible by Nstar
#calc_error(M,N,filename,'t')  

### Unsure if the refinements below should be included.

# ---| r -refinement, here both M and N increases. |--- #
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([8,10,20,25,40,50,100,125,200,250,500])
#calc_error(M,N,filename,'r')  #gives BE; Ndof^(-1/2) and CN; Ndof^(-1)

# r -refinement, keeping r fixed, r=40=M^2/N. Difficult to choose appropriate values
M = np.array([20,25,40,50,100,125,200])
N = np.array([10,16,40,63,250,391,1000]) 
#calc_error(M,N,filename,'r')  #gives something weird :=O
