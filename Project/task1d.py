"""Numerical solution using the given difference method in task 1d."""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def f(x,epsilon):
    """Right hand side of 1D Poisson equation."""
    return epsilon**(-2)*np.exp(-(1/epsilon)*(x-0.5)**2)*(4*x**2 - 4*x + 1 - 2*epsilon)


def anal_solution(x, epsilon):
    """Manufactured solution of the Poisson equation with Dirichlet BC"""
    return np.exp(-(1/epsilon)*(x-0.5)**2)


def num_solution(x, M, epsilon): #This is a second order method, we also need a first order method
    """Numerical solution of the Possion equation with Dirichlet BC. given by the manufactured solution"""

    h = 1/(M+1)

    #Construct Dirichlet boundary condition from manuactured solution
    alpha = anal_solution(0,epsilon)
    beta = anal_solution(1,epsilon)
    
    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2

    # Construct f.
    f_vec = np.full(M, f(x[1:-1],epsilon))
    f_vec[0] = f_vec[0] - alpha/h**2
    f_vec[-1] = f_vec[-1] - beta/h**2

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    
    return Usol

##Starting on AMR

def coeff_stencil(i,x): #i can go from i=1 to i=M
    h = np.zeros(len(x)-1)
    h[:] = x[1:] - x[:-1]
    if i==1:
        d_p = h[1]
        d_m = h[0]
        d_2m = 2*h[0] #Setter denne verdien (vet egt. ikke h_-1)
    else:
        d_p = h[i]
        d_m = h[i-1]
        d_2m = h[i-2] + h[i-1]
    a = 2 * (d_p - d_m) / (d_2m*(d_2m + d_p)*(d_2m - d_m))
    b = 2 * (d_2m - d_p) / (d_m*(d_2m - d_m)*(d_m + d_p))
    c = 2 * (d_2m + d_m) / (d_p*(d_m + d_p)*(d_2m + d_p))
    return a, b, c


def num_solution_four_point_stencil(x):
    '''Makes the matrix Ah, the discretizaion of U_xx, depentent on the grid x'''
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,x)
        
    data = np.array([a, b, -(a+b+c), c])
    diags = np.array([-2,-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()
    
    alpha = anal_solution(0,epsilon)
    beta = anal_solution(1,epsilon)
    
    f_vec = np.full(M, f(x[1:-1],epsilon))
    f_vec[0] = f_vec[0] - b[0]*alpha
    f_vec[1] = f_vec[1] - a[1]*alpha
    f_vec[-1] = f_vec[-1] - c[-1]*beta

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    return Usol

'''
def check():
    """Encapsulate the plotting under development for ease of use."""
    plt.plot(x, num_solution(x, M, epsilon), label="Num", color = "red")
    plt.plot(x, anal_solution(x, epsilon), label="An", color = "black", linestyle = "dotted")
    plt.plot(x,num_solution_four_point_stencil(x),label="AMR",color="green")
    plt.legend()
    plt.show()
    # Looks good!
'''

def AMR_average(x0,steps):  #using average
    '''Finds error with four point stencil solution, here for disc. norm'''
    disc_error = np.zeros(steps)
    Usol_M = [num_solution_four_point_stencil(x0)] #using regular lists so we can append arrays of different shapes
    X_M = [x0]
    for k in range(steps-1):
        
        u = anal_solution(X_M[-1],epsilon)        
        error = la.norm(Usol_M[-1] - u)
        disc_error[k] = error/la.norm(u) #relative error
        average_error = error/len(X_M[-1])
        
        #refine the grid
        x = np.copy(X_M[-1])
        n = 0
        for i in range(1,len(Usol_M[-1])-1): #know the correct values at the boundary
            if abs(Usol_M[-1][i]-u[i]) > average_error:
                x = np.insert(x,i+n,(X_M[-1][i]+X_M[-1][i-1])/2)
                n += 1
        X_M.append(x)
        Usol_M.append(num_solution_four_point_stencil(x))
        
    u = anal_solution(X_M[-1],epsilon)
    disc_error[-1] = la.norm(Usol_M[-1]-u)/la.norm(u)
    return Usol_M,X_M,disc_error
    
M = 4
x = np.linspace(0, 1, M+2) # Make 1D grid.
epsilon = 0.01
steps = 5

U, X, disc_error = AMR_average(x,steps)
print(X[0],X[-1])
#plt.plot([i+1 for i in range(steps)],disc_error)
#plt.show()
for i in range(steps):
    plt.plot(X[i],U[i],label=str(i))

plt.plot(X[-1],anal_solution(X[-1],epsilon),label="An",linestyle='dotted')
plt.legend()
plt.show()      ### Something is terribly wrong! :/
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
    Usol = num_solution(x, M = m, alpha = alpha, sigma = sigma)
    analsol = anal_solution(x, alpha, sigma)
    discrete_error[i], discrete_errorF[i] = e_l(Usol, analsol)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(M, discrete_error, label="el", color = "red")
ax.plot(M, discrete_errorF, label="F", color = "black", linestyle = "dotted")
plt.legend()
plt.grid() # Looks like it decreases with two orders!
plt.show() 
'''