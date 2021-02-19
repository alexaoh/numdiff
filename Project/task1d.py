
from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

epsilon = 0.01
def f(x):
    """Right hand side of 1D Poisson equation."""
    return epsilon**(-2)*np.exp(-(1/epsilon)*(x-0.5)**2)*(4*x**2 - 4*x + 1 - 2*epsilon)


def anal_solution(x):
    """Manufactured solution of the Poisson equation with Dirichlet BC"""
    return np.exp(-(1/epsilon)*(x-0.5)**2)


def num_sol_second_order(x, M): #This is a second order method, using central difference
    """Second order numerical solution of the Possion equation with Dirichlet BC. given by the manufactured solution"""

    h = 1/(M+1)

    #Construct Dirichlet boundary condition from manuactured solution
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2

    # Construct f.
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - alpha/h**2
    f_vec[-1] = f_vec[-1] - beta/h**2

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    
    return Usol

def num_sol_first_order(x,M):
    """First order numerical solution of the Possion equation with Dirichlet B.C. given by the manufactured solution
        Using a forward difference scheme"""
    h = 1/(M+1)

    #Construct Dirichlet boundary condition from manuactured solution
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    f_vec = np.full(M,f(x[:-2]))
    f_vec[0] = f_vec[0]- alpha/h**2
    f_vec[-1] = f_vec[-1] - beta/h**2

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    return Usol

M = 20
x = np.linspace(0, 1, M+2) # Make 1D grid.

Usol_1 = num_sol_first_order(x,M)
Usol_2 = num_sol_second_order(x,M)
plt.plot(x,Usol_1,label="num1")
plt.plot(x,Usol_2,label="num2")
plt.plot(x,anal_solution(x),label="an")
plt.legend()
plt.show()

### Uniform mesh refinement UMR, here using disc_error
M_list = np.linspace(20,1000,20,dtype=int)
h_list = 1/(M_list+1)

e_1 = np.zeros(len(M_list))
e_2 = np.zeros(len(M_list))
for i in range(len(M_list)):
    x = np.linspace(0,1,M_list[i]+2)
    u = anal_solution(x)
    e_1[i] = la.norm(num_sol_first_order(x,M_list[i]) - u)/la.norm(u)
    e_2[i] = la.norm(num_sol_second_order(x,M_list[i]) - u)/la.norm(u)

plt.plot(M_list,e_1,label="l2_first")
plt.plot(M_list,8*h_list,label='O(h)',linestyle='dashed')
plt.plot(M_list,10*h_list**2,label="O(h^2)",linestyle='dashed')
plt.plot(M_list,e_2,label="l2_second")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


##---------Starting on AMR--------------
def coeff_stencil(i,h): #i can go from i=1 to i=M
    if i==1:
        d_2m = 2*h[0] #Setter denne verdien (vet egt. ikke h_-1)
    else:
        d_2m = h[i-2] + h[i-1]
    d_p = h[i]
    d_m = h[i-1]
    a = 2 * (d_p - d_m) / (d_2m*(d_2m + d_p)*(d_2m - d_m))
    b = 2 * (d_2m - d_p) / (d_m*(d_2m - d_m)*(d_m + d_p))
    c = 2 * (d_2m + d_m) / (d_p*(d_m + d_p)*(d_2m + d_p))
   
    return a, b, c


def num_solution_four_point_stencil(x):
    '''Makes the matrix Ah, the discretizaion of U_xx, depentent on the grid x'''
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    h = np.zeros(len(x)-1)
    h[:] = x[1:] - x[:-1]
    
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,h)

    data = np.array([a, b, -(a+b+c), c])
    diags = np.array([-2,-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()
    
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - b[0]*alpha
    #f_vec[1] = f_vec[1] - a[1]*alpha #Should this be here?
    f_vec[-1] = f_vec[-1] - c[-1]*beta

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    return Usol


def AMR_average(x0,steps):  #using average
    '''Uses mesh refinement 'steps' times. Finds the error, x-grid and numerical solution for each step.'''
    disc_error = np.zeros(steps+1)
    Usol_M = [num_solution_four_point_stencil(x0)] #using regular lists so we can append arrays of different shapes
    X_M = [x0]
    for k in range(steps):
        
        u = anal_solution(X_M[-1])        
        error = la.norm(Usol_M[-1] - u)
        disc_error[k] = error/la.norm(u) #relative error
        average_error = error/len(X_M[-1]) #is this the correct form?
        
        #refine the grid
        x = np.copy(X_M[-1])
        n = 0 #hjelpevariabel
        for i in range(1,len(Usol_M[-1])-1): #know the correct values at the boundary
            if abs(Usol_M[-1][i]-u[i]) > average_error:
                x = np.insert(x,i+n,(X_M[-1][i]+X_M[-1][i-1])/2)
                n += 1
        X_M.append(x)
        Usol_M.append(num_solution_four_point_stencil(x))
        
    u = anal_solution(X_M[-1])
    disc_error[-1] = la.norm(Usol_M[-1]-u)/la.norm(u)
    return Usol_M, X_M, disc_error



steps = 3
'''
#Test with a plot
U, X, disc_error = AMR_average(x,steps)
for i in range(steps+1):
    plt.plot(X[i],U[i],label=str(i))

plt.plot(X[-1],anal_solution(X[-1],epsilon),label="An",linestyle='dotted')
plt.legend()
plt.show()      ### Think it is somethin wrong with the stencil, cant see what excactly
'''