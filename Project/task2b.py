
'''
Solves u_t = u_xx on x[0,1], t[0,T] with reference to manufactured solution with UMR with r-refinement.
Dirichlet BC u(0,t)=u(1,t)=0, and initial value f(x)=3*sin(2*pi*x)
First order method; uses Eulers method (forward)
Second order; Crank Nicolson
'''

from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.linalg import toeplitz, solve_toeplitz
import numpy as np
import numpy.linalg as la
from scipy.interpolate import interp1d 
from scipy.integrate import quad

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


def cont_L2_norm(v, t):
    """Continuous L2 norm of v(x) between left and right. """
    integrand = lambda x: v(x,t)**2
    return np.sqrt(quad(integrand, 0, 1)[0])

def e_L(U, u, t):
    """Relative error e_L.
    
    U: Approximate numerical solution.
    u: Analytical solution. 
    """
    f = lambda x,t : u(x,t) - U(x)
    numer = cont_L2_norm(f,t)
    denom = cont_L2_norm(u,t)

    return numer/denom

initial = (lambda x: 3*np.sin(2*np.pi*x))

def anal_solution(x,t):
    return 3*np.exp(-4*(np.pi**2)*t)*np.sin(2*np.pi*x)

def theta_method(x, t, theta):
    """Theta-method.
    M: Number of internal points in x-dimension in the grid. 
    N: Number of internal points in t-direction in the grid.  
    theta: Parameter to change how the method works. 
    
    Returns a new list X with the solution of the problem in the gridpoints. 
    """
    M = len(x)-2
    N = len(t)-1
    U = np.zeros((N+1,M+2))
    U[0,:] = initial(x)
    U[:,0] = 0
    U[:,-1] = 0
    
    # Calculate step lengths. 
    h = x[1] - x[0]
    k = t[1] - t[0]
    r = k/h**2
    
    # Set up the array Ah with scipy.sparse.spdiags, 
    #data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    #diags = np.array([-1, 0, 1])
    #Ah = spdiags(data, diags, M, M).toarray()
    
    col = np.zeros(M)
    col[0] = -2
    col[1] = 1
    Ah = toeplitz(col)
    b = (np.identity(M) +(1-theta)*r*Ah)
    
    lhs = np.zeros(M)
    lhs[0] = 1+2*theta*r
    lhs[1] = -theta*r
    for n in range(N):
        U[n+1,1:-1] = solve_toeplitz(lhs,b @ U[n,1:-1])
        
        #U[n+1,1:-1] = la.solve((np.identity(M)-theta*r*Ah),RHS)
        #U[n+1,1:-1] = spsolve((np.identity(M)-theta*r*Ah),RHS)
    return U

#x = np.linspace(0,1,M+2)
'''
U_BE = theta_method(x,t,1)
U_CN = theta_method(x,t,1/2)

plt.plot(x,U_CN[-1,:],label='CN')
plt.plot(x,U_BE[-1,:],label='BE')
plt.plot(x,anal_solution(x,t[-1]),label='Analytic')  
plt.legend()
plt.show()
'''

# Choose time t[-1]=T=0.2, as the time to look at errors.

def plot_UMR(M,N,type): # type = 't', 'h' or 'r'-refinement
    T = 0.2
    time_index = -1
    refine_list = []
    if type=='h':
        refine_list = M
        N = np.ones_like(M)*N
    elif type=='t':
        refine_list = N
        M = np.ones_like(N)*M
    else:
        refine_list = M
    disc_err_first = np.zeros(len(refine_list))
    disc_err_second = np.zeros(len(refine_list))
    cont_err_first = np.zeros(len(refine_list))
    cont_err_second = np.zeros(len(refine_list))
    for i in range(len(refine_list)):
        x = np.linspace(0,1,M[i]+2) 
        t = np.linspace(0,T,N[i]+1)
        
        U_BE = theta_method(x,t,1)
        U_CN = theta_method(x,t,1/2)
        u = anal_solution(x,t[time_index])
        
        disc_err_first[i] = e_l(U_BE[time_index,:],u)
        disc_err_second[i] = e_l(U_CN[time_index,:],u)

        cont_err_first[i] = e_L(interp1d(x, U_BE[time_index,:], kind = 'cubic'), anal_solution, t[time_index])
        cont_err_second[i] = e_L(interp1d(x, U_CN[time_index,:], kind = 'cubic'), anal_solution, t[time_index])
    
    MN = M*N    
    Ndof = 1/(MN)
    plt.plot(MN,cont_err_first, label=r"$e^r_{L_2}$ BE", color='green',marker='o', linewidth = 1.4)
    plt.plot(MN,cont_err_second, label=r"$e^r_{L_2}$ CN",color='purple',marker='o', linewidth = 1.4)
    plt.plot(MN, disc_err_first, label=r"$e^r_{l}$ BE", color='red',marker='o',linestyle="--", linewidth = 1)
    plt.plot(MN, disc_err_second, label=r"$e^r_{l}$ CN",color='orange',marker='o',linestyle="--", linewidth = 1)
    
    # These changes for different methods, change it manually!!
    #plt.plot(MN, (2e+3)*Ndof, label=r"O$(N_{dof}^{-1})$", linestyle='dashed', color='red')
    plt.plot(MN, (1e+3)*Ndof**(2/3), label=r"O$(N_{dof}^{-2/3})$",linestyle='dashed', color='blue', linewidth = 1)
    plt.plot(MN, (2e+1)*Ndof**(2/3),linestyle='dashed', color='blue', linewidth = 1)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M*N$')
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.legend()
    plt.grid()
    plt.show()

# h-refinement
N = 1000
M = np.array([8,16,32,64,128,256]) 
#plot_UMR(M,N,'h')  # h-refinement, both methods are second order but BE flattens out because of big error in t

# t-refinement
M = 1000
N = np.array([8,16,32,64,128,256])
#plot_UMR(M,N,'t') #t-refinement, BE går som O(h), CN som O(h^2)

# r-refinement, double N and M for each step
M = np.array([8,16,32,64,128,256])
N = np.array([8,16,32,64,128,256])
#plot_UMR(M,N,'r')  #BE går som O(h^1/2), CN som O(h^1)

# r-refinement with constant r, here r=1024=k/h^2=M^2/N
M = np.array([64,128,256,512,1024,2048])
N = np.array([4,16,64,256,1024,4096])
plot_UMR(M,N,'r')  #BE og CN går som O(h^2/3) (etterhvert i hvert fall)


