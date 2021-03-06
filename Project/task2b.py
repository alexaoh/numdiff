
'''
Solves u_t = u_xx on x[0,1], t[0,T] with reference to manufactured solution, both with UMR and AMR and both first and second order
Here we have Dirichlet BC u(0,t)=u(1,t)=0, and initial value f(x)=3*sin(2*pi*x)
First order method; uses Eulers method (forward)
Second order; Crank Nicolson
'''

from plot_heat_eqn import *
from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.linalg import toeplitz, solve_toeplitz
import numpy as np
import numpy.linalg as la

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

from scipy.interpolate import interp1d 
from scipy.integrate import quad

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
    
    lhs = np.zeros(M)
    lhs[0] = 1+2*theta*r
    lhs[1] = -theta*r
    for n in range(N):
        b = (np.identity(M) +(1-theta)*r*Ah) @ U[n,1:-1]
        U[n+1,1:-1] = solve_toeplitz(lhs,b)
        
        #U[n+1,1:-1] = la.solve((np.identity(M)-theta*r*Ah),RHS)
        #U[n+1,1:-1] = spsolve((np.identity(M)-theta*r*Ah),RHS)
        
    return U

#x = np.linspace(0,1,M+2)
'''
U_BE = theta_method(x,t,1)
U_CN = theta_method(x,t,1/2)

plt.plot(x,U_CN[-1,:],label='CN')
plt.plot(x,U_BE[-1,:],label='BE')
plt.plot(x,anal_solution(x,t[-1]),label='Anal')  
plt.legend()
plt.show()
'''

#choose time t[-1]=T=0.2, as the time to look at errors.
T = 0.2
time_index = -1

num = 7
M = np.linspace(10,500,num,dtype=int) 


r0 = 4 # this is the relation we want to keep, we can maybe increase to >1 if N gets too large
# uses k=r0*h^2 and N= T/k - 1 to define a new t-grid dependent on the x-grid

# ------ Problems ------- 
# 1) it takes a looong time to run with this relation for large M>400. N gets way too large, 
# suggestion; have a linear increase in r0 for each iteration e.g r0 e [1,3]
# 2) Backward Euler seems to go as O(h^2), is this because we use central difference along x-axis. 
# Maybe we need to keep M constant and just refine the t-axis and plot error as a function of N? - tried this, did not work!
    
def plot_error_UMR(M):
    disc_err_first = np.zeros(num)
    disc_err_second = np.zeros(num)
    cont_err_first = np.zeros(num)
    cont_err_second = np.zeros(num)
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+2)
        h = x[1] - x[0]
        
        #make a new t-grid based on r0 and M (or h if you like)
        N = int(T/(r0*h**2)) - 1  
        t = np.linspace(0,T,N+1)  
        print(i, (t[1]-t[0])/h**2, M[i], N)
        
        U_BE = theta_method(x,t,1)
        U_CN = theta_method(x,t,1/2)
        u = anal_solution(x,t[time_index])
        
        disc_err_first[i] = la.norm(U_BE[time_index,:]-u)/la.norm(u)
        disc_err_second[i] = la.norm(U_CN[time_index,:]-u)/la.norm(u)
    
        #continous error
        #cont_err_first[i] = e_L(interp1d(x, U_BE[time_index,:], kind = 'cubic'), anal_solution, t[time_index])
        #cont_err_second[i] = e_L(interp1d(x, U_CN[time_index,:], kind = 'cubic'), anal_solution, t[time_index])
    
    h = 1/(M+2)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(M, disc_err_first, label='l2_first', color='red')
    plt.plot(M, disc_err_second, label='l2_second',color='blue')
    #plt.plot(M,cont_err_first, label='L2_first')
    #plt.plot(M,cont_err_second, label='L2_second')
    plt.plot(M, 20*h, label='O(h)', linestyle='dotted', color='red')
    plt.plot(M, 50*h**2, label='O(h^2)',linestyle='dotted', color='blue')
    plt.legend()
    plt.grid()
    plt.show()
    
#plot_error_UMR(M)



## ----- AMR -----
from scipy.sparse import diags # Make sparse matrices with scipy.

def first_order_AMR(x,t):

    M = len(x)-2
    N = len(t)-1
    U = np.zeros((N+1,M+2))
    U[0,:] = initial(x)
    U[:,0] = 0
    U[:,-1] = 0
    
    h = x[1:] - x[:-1]
    k = t[1:] - t[:-1]
    
    b = 2/(h[:-1]*(h[:-1]+h[1:]))    
    c = 2/(h[1:]*(h[1:] + h[:-1]))

    data = [b[1:], -(b+c), c[:-1]]   #how does the B.C fit in here?
    diagonals = np.array([-1, 0, 1])
    Ah = diags(data, diagonals).toarray()

    for n in range(N):  #Backward Euler
        b = U[n,1:-1]
        lhs = np.identity(M)-k[n]*Ah
        U[n+1,1:-1] = la.solve(lhs,b)
    return U


def AMR(x0,t0,steps):
    disc_error = np.zeros(steps+1)
    M_list = np.zeros(steps+1)
    N_list = np.zeros(steps+1)
    Usol_M = [first_order_AMR(x0,t0)]
    X_M = [x0]
    T_M = [t0]
    for k in range(steps):
        M_list[k] = len(X_M[-1])-2
        N_list[k] = len(T_M[-1])-1

        x_1,t_1 = np.meshgrid(X_M[-1],T_M[-1])
        u = anal_solution(x_1,t_1)
        
        
        #Using the whole grid to refine x and t
        disc_error[k] = la.norm(Usol_M[-1]-u)/la.norm(u)
        
        #not sure how to determine the error for x- and t-axis. Suggestion is below. 
        #x-refinement
        error_ave_x = np.average(np.abs(Usol_M[-1]-u),axis=0)
        e_x = (error_ave_x[1:] + error_ave_x[:-1])/2
        error_ave = np.average(error_ave_x)
        
        x = np.copy(X_M[-1])
        n = 0
        for i in range(len(e_x)):
            if e_x[i] > error_ave:
                x = np.insert(x,i+n+1,(X_M[-1][i]+X_M[-1][i+1])/2)        
                n += 1
        
        #t-refinement
        error_ave_t = np.average(np.abs(Usol_M[-1]-u),axis=1)
        e_t = (error_ave_t[1:] + error_ave_t[:-1])/2
        error_ave = np.average(error_ave_t)
        
        t = np.copy(T_M[-1])
        n = 0
        for i in range(len(e_t)):
            if e_t[i] > error_ave:
                t = np.insert(t,i+n+1,(T_M[-1][i]+T_M[-1][i+1])/2)        
                n += 1
                
        X_M.append(x)
        T_M.append(t)
        Usol_M.append(first_order_AMR(x,t))

    #add last elements
    x_1,t_1 = np.meshgrid(X_M[-1],T_M[-1])
    u = anal_solution(x_1,t_1)
    disc_error[-1] = la.norm(Usol_M[-1]-u)/la.norm(u)
    M_list[-1] = len(X_M[-1])-2
    N_list[-1] = len(T_M[-1])-1
    return Usol_M, X_M, T_M, disc_error, M_list, N_list
        
M = 9
N = 15

x = np.linspace(0,1,M+2)
t = np.linspace(0,T,N+1)

steps = 7

U , X, T, disc_error, Mlist, Nlist = AMR(x,t,steps)

for i in range(steps):
    k = np.average(T[i][1:]-T[i][:-1])
    h = np.average(X[i][1:]-X[i][:-1])
    print(k/h**2)
    
plt.plot(Mlist,disc_error)
plt.xscale('log')
plt.yscale('log')
plt.show()
    