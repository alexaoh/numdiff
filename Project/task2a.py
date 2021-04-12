#from crank_nicolson import * # is this needed? Delete the file and this comment if not ;)
from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import pickle # To save the reference solution.

def plot_order(Ndof, error_start, order, label, color):
    const = error_start/Ndof[0]**order
    plt.plot(1/Ndof, const*Ndof**order, label=label, color=color, linestyle='dashed')


def disc_l2_norm(V):
    """Discrete l2-norm of V."""
    sqr = (lambda x: x**2) 
    return np.sqrt(1/len(V)*sum(list(map(sqr, V))))

def e_l(U, u):
    """Relative error e_l.

    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    return disc_l2_norm(u-U)/disc_l2_norm(u) # la.norm(u-U)/la.norm(u)

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

def calcSol(x, t, order, theta, plot = False):
    """ Hva gjør denne egentlig? Skriv her ;)
    order = 1: Use bd and fd on bc + trapezoidal.
    order = 2: Use central differences with fict. nodes on bc + trapezoidal.
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
    sol = theta_method_given_Q(x, t, V0, Q, theta) # Q: Is this the right form, using BE and CN for first and second order?
  
    if plot:
        tv, xv = np.meshgrid(t,x)
        three_dim_plot(xv = xv, tv = tv, I = sol.T, label = "Numerical Solution")
    
    return sol


def saveRefSol(Mstar, Nstar, order, filename):
    """Save the reference solution to file."""
    #refSol = calcSol(Mstar, order, plot = False)
    T = 0.2
    x = np.linspace(0,1,Mstar+1)
    t = np.linspace(0,T,Nstar+1)
    refSol = calcSol(x, t, order) # Tror det mangler noen argumenter her muligens.
    with open(filename, 'wb') as fi:
        pickle.dump(refSol, fi)


def calcError(M,N,filename,typ):  #type = h,t,r - refinement. M,N are either lists or scalars depending on type of refinement
                                  ##### Ta bort disse kommentarene hvis nok info står i docstring nedenfor.
    """Calculate the relative error with the reference solution.
    
    Input:
    M: List of ...
    N: List of ...
    filename: Name of the file where the reference solution has been saved. 
    type: h or t ...
    """
    assert(typ == "t" or typ == "r") # SKulle det vært noe r - refinement her?
    refSol = None
    Mstar = Nstar = 1000   # Change values here if you change it in the file.
    with open(filename, 'rb') as fi: 
        refSol = pickle.load(fi)

    T = 0.2  
    refTime = -1 
    if typ=='h':
        N = np.ones_like(M)*N
    elif typ=='t':
        M = np.ones_like(N)*M
    modulus = Mstar % M    #controls that M are divisible by Mstar. This does not apply to N because we look at error in T which is similar for both sol and refSol
    if len(modulus[np.nonzero(modulus)]) != 0:
        print('Wrong M values.')
        return 1
        
    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))

    # Piecewise constant function uStar.
    # piecewise = lambda Ustar, M, Mstar : [Ustar[math.floor(i*(Mstar + 1)/(M + 1))] for i in range(M + 1)].
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+1)
        t = np.linspace(0,T,N[i]+1)
        sol_1 = calcSol(x, t, 2, 1)
        sol_2 = calcSol(x, t, 2, 1/2)
        #uStar = np.array(piecewise(refSol[refTime,:], M, Mstar))
        uStar = refSol[refTime,0::(Mstar//M[i])]
       
        disc_err_first[i] = e_l(sol_1[refTime,:], uStar)
        disc_err_second[i] = e_l(sol_2[refTime,:], uStar)      
    
    MN = M*N
    Ndof = 1/MN
    plt.plot(MN, disc_err_first, label = "$e^r_l$ (BE)",color='red',marker='o')
    plt.plot(MN, disc_err_second, label = "$e^r_l$ (CN)",color='blue',marker='o')
    #plt.plot(MN, (3e-1)*Ndof**(1/2), label="$O(N_{dof}^{-1/2})$", color='red', linestyle='dashed') 

    plot_order(Ndof, disc_err_first[0], 1/2, label = "$O(N_{dof}^{-1/2})$", color = 'red')
    plot_order(Ndof, disc_err_second[0], 1, label = "$O(N_{dof}^{-1})}$", color='blue')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('$M*N$')
    plt.ylabel(r"Error $e^r_{l}$")
    plt.legend()
    plt.grid()
    plt.show()

# Kan rydde litt nedenfor også kanskje?

Mstar = Nstar = 1000
filename = 'refSol.pk'

#saveRefSol(Mstar, Nstar, 2, filename) # Only needs to be run once, or if you change Mstar
## ObsObs, here I have used refSol with order 2 for comparing with both orders, is this wrong or just more accurate?

# h -refinement
N = 1000
M = np.array([8,10,20,25,40,50,100,125,200,250,500])

#calcError(M,N,filename,'h')

# t - refinement
M = 1000
#N = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([4,8,16,32,64,128,256])  #does not have to be divisible by Nstar
#calcError(M,N,filename,'t')  #why does the first order method give a much lower error at last point (N=500)?

# r -refinement, here both M and N increases.
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([8,10,20,25,40,50,100,125,200,250,500])
calcError(M,N,filename,'r')  #gives BE; Ndof^(-1/2) and CN; Ndof^(-1)

# r -refinement, keeping r fixed, r=40=M^2/N. Difficult to choose appropriate values
M = np.array([20,25,40,50,100,125,200])
N = np.array([10,16,40,63,250,391,1000]) 
#calcError(M,N,filename,'r')  #gives something weird :=O
