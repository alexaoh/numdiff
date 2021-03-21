from scipy.sparse import spdiags,diags # Make sparse matrices with scipy.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import *
from scipy.interpolate import interp1d 

epsilon = 0.01
def f(x):
    """Right hand side of 1D Poisson equation."""
    return epsilon**(-2)*np.exp(-(1/epsilon)*(x-0.5)**2)*(4*x**2 - 4*x + 1 - 2*epsilon)


def anal_solution(x):
    """Manufactured solution of the Poisson equation with Dirichlet BC."""
    return np.exp(-(1/epsilon)*(x-0.5)**2)


def num_sol_UMR(x,M,order): #order = 1 or 2
    """First order numerical solution of the Possion equation with Dirichlet B.C.,
    given by the manufactured solution. Using a forward difference scheme.
    """
    h = 1/(M+1)

    # Construct Dirichlet boundary condition from manuactured solution.
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    if order == 1:
        f_vec = np.full(M,f(x[:-2]))
    else: #order = 2
        f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0]- alpha/h**2
    f_vec[-1] = f_vec[-1] - beta/h**2

    # Solve linear system. 
    Usol = la.solve(Ah, f_vec)
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol,beta)
    
    return Usol

### Uniform mesh refinement UMR
M_list = np.linspace(20,1000,20,dtype=int)
h_list = 1/(M_list+1)

e_1_disc = np.zeros(len(M_list))
e_2_disc = np.zeros(len(M_list))
e_1_cont = np.zeros(len(M_list))
e_2_cont = np.zeros(len(M_list))

for i, m in enumerate(M_list):
    x = np.linspace(0,1,m+2)
    u = anal_solution(x)
    first_order_num = num_sol_UMR(x,m,1)
    second_order_num = num_sol_UMR(x,m,2)

    # Discrete norms. 
    e_1_disc[i] = e_l(first_order_num, u)
    e_2_disc[i] = e_l(second_order_num, u)
    
    # Continuous norms. 
    interpU_first = interp1d(x, first_order_num, kind = 'cubic')
    interpU_second = interp1d(x, second_order_num, kind = 'cubic')

    e_1_cont[i] = e_L(interpU_first, anal_solution, x[0], x[-1])
    e_2_cont[i] = e_L(interpU_second, anal_solution, x[0], x[-1])

def plot_errors_UMR(save = False):
    """Encapsulation, to avoid commenting while development."""
    plt.plot(M_list,e_2_cont,label=r"$e^r_{L_2}$ second", color = "yellow",marker='o')
    plt.plot(M_list,e_1_cont,label=r"$e^r_{L_2}$ first", color = "green",marker='o')
    plt.plot(M_list,e_1_disc,label=r"$e^r_{l}$ first", color = "red",marker='o',linestyle = "--")
    plt.plot(M_list,e_2_disc,label=r"$e^r_l$ second", color = "blue",marker='o',linestyle = "--")
    plt.plot(M_list,7*h_list,label=r"O$(h)$",linestyle='dashed', color = "red")
    plt.plot(M_list,10*h_list**2,label=r"O$(h^2)$",linestyle='dashed', color = "blue")
    
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel("Number of points $M$")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("loglogtask1dUMR.pdf")
    plt.show()

#plot_errors_UMR()

##--------- AMR--------------
def coeff_stencil(i,h):
    """Calculates the coefficients in the four-point stencil for a given index i and list h."""
    #d_p, d_m, d_2m = d_i+1, d_i-1, d_i-2 (notation from the article)
    if i==1:    #use a 3-point stencil with equal spacing at first iteration, that means h[0]=h[1].
        b = c = 1/h[0]**2
        a = 0
    else:
        d_2m = h[i-2] + h[i-1]
        d_p = h[i]
        d_m = h[i-1]
        a = 2 * (d_p - d_m) / (d_2m*(d_2m + d_p)*(d_2m - d_m)) 
        b = 2 * (d_2m - d_p) / (d_m*(d_2m - d_m)*(d_m + d_p))
        c = 2 * (d_2m + d_m) / (d_p*(d_m + d_p)*(d_2m + d_p))
    return a, b, c


def num_sol_AMR_second(x):
    """Second order discretization of U_xx with nonuniform grid, using the four-point stencil."""
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    h = np.diff(x)
    
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,h)  # a[0] = a_1 in the scheme
    
    data = [a[2:], b[1:], -(a+b+c), c[:-1]]
    diagonals = np.array([-2,-1, 0, 1])  
    Ah = diags(data, diagonals).toarray()  #Note; Here using sparse.diags, not sparse.spdiags!
    
    alpha = anal_solution(0)
    beta = anal_solution(1)
    
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - b[0]*alpha
    f_vec[1] = f_vec[1] - a[1]*alpha
    f_vec[-1] = f_vec[-1] - c[-1]*beta

    Usol = la.solve(Ah, f_vec)
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol,beta)
    
    return Usol  


def num_sol_AMR_first(x):
    """First order discretization of U_xx with nonuniform grid, using the three-point stencil."""
    M = len(x)-2
    h = np.diff(x)
 
    b = 2/(h[:-1]*(h[:-1]+h[1:]))    
    c = 2/(h[1:]*(h[1:] + h[:-1]))
    
    data = [b[1:], -(b+c), c[:-1]]   
    diagonals = np.array([-1, 0, 1])
    Ah = diags(data, diagonals).toarray()
        
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - alpha*b[0]
    f_vec[-1] = f_vec[-1] - beta*c[-1]

    Usol = la.solve(Ah,f_vec)
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol, beta)
    
    return Usol

def calc_cell_errors(u,U):
    '''Calulculates an error for each cell by taking the avg of the error in the endpoints.'''
    n = len(u) - 1 # Number of cells
    cell_errors = np.zeros(n)
    for i in range(n):
        cell_errors[i] = abs(U[i] - u[i])
    return cell_errors


def AMR(x0, steps, num_solver): 
    """Uses mesh refinement 'steps' times. Finds the error, x-grid and numerical solution for each step."""
    disc_error = np.zeros(steps+1)
    cont_error = np.zeros(steps+1)
    M_list = np.zeros(steps+1)
    Usol_M = [num_solver(x0)]
    X_M = [x0]
    for k in range(steps):
        M_list[k] = len(X_M[-1])-2
        u = anal_solution(X_M[-1])        
        disc_error[k] = e_l(Usol_M[-1],u)
         
        interpU = interp1d(X_M[-1], Usol_M[-1], kind = 'cubic')
        cont_error[k] = e_L(interpU, anal_solution, 0, 1)

        x = list(np.copy(X_M[-1]))
        cell_errors = calc_cell_errors(u, Usol_M[-1])
        tol = 1 * np.average(cell_errors)
       
        j = 0 # Index for x in case we insert points

        # For testing if first or second cell have been refined
        firstCell = False
        for i in range(len(cell_errors)):
            if cell_errors[i] > tol:
                x.insert(j+1, x[j] + 0.5 * (x[j+1] - x[j]))
                j += 1
                
                # Tests to ensure that first and second cell have same length
                if i == 0:
                    firstCell = True
                    x.insert(j+1, x[j] + 0.5 * (x[j+1] - x[j]))
                    j += 1
                if i == 2 and not firstCell:
                    x.insert(1, x[0] + 0.5 * (x[1] - x[0]))
                    j += 1
            j += 1
        
        x = np.array(x)    
        X_M.append(x)
        Usol_M.append(num_solver(x))
        
    #add last elements
    u = anal_solution(X_M[-1])
    disc_error[-1] = e_l(Usol_M[-1],u)
    interpU = interp1d(X_M[-1], Usol_M[-1], kind = 'cubic')
    cont_error[-1] = e_L(interpU, anal_solution, 0, 1)
    M_list[-1] = len(X_M[-1])-2
    return Usol_M, X_M, disc_error, cont_error, M_list


def test_plot_AMR_solver(num_solver):
    """Show plots of numerical and analytical solution."""
    M = 9
    x = np.linspace(0, 1, M+2)
    steps = 4
    U, X, _, _ = AMR(x,steps,num_solver)
    for i in range(0,steps+1):
        plt.plot(X[i],U[i],label=str(i))

    plt.plot(X[-1],anal_solution(X[-1]),label="An",linestyle='dotted')
    plt.legend()
    plt.show()
#test_plot_AMR_solver(num_sol_AMR_second)


#---plot errors---
M = 9
x0 = np.linspace(0, 1, M+2)
steps = 16

U_1, X_1, disc_error_1, cont_error_1, M_1 = AMR(x0,steps,num_sol_AMR_first)
U_2, X_2, disc_error_2, cont_error_2, M_2 = AMR(x0,steps,num_sol_AMR_second)


def plot_bar_error(X,U,start,stop):
    """Plot error at each cell as bar-plot"""
    if stop > steps + 1:
        return 0
    
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(15,15))
    for i in range(start,stop):    

        h = X[i][2:] - X[i][1:-1]
        u = anal_solution(X[i])   
        cell_errors = calc_cell_errors(u, U[i])
        tol = 1 * np.average(cell_errors)

    
        axs.flatten()[i].plot([j for j in range(len(cell_errors))], [tol for j in range(len(cell_errors))],label='ave',linestyle='dashed') #average AMR
        axs.flatten()[i].bar([j for j in range(len(cell_errors))], cell_errors, align='edge',label=str(i))
        #Question; why plot with whole numbers as x-axis and not the x-grid we are refining for each step (X[i])?
    
    plt.legend()
    fig.tight_layout()
    plt.show()

#plot_bar_error(X_2,U_2,0,steps)


def plot_AMR_errors(save=False):
    h = 1/(M_2+1)
    plt.plot(M_1, disc_error_1, label="$e_l^r$ (3 point stencil)",color='red',marker='o',linewidth=2)
    plt.plot(M_1, cont_error_1, label="$e_L^r$ (3 point stencil)",color='red',linestyle="--",marker='o',linewidth=2)
    plt.plot(M_2, disc_error_2, label="$e_l^r$ (4 point stencil)",color='blue',marker='o',linewidth=2)
    plt.plot(M_2, cont_error_2, label="$e_L^r$ (4 point stencil)",color='blue',linestyle="--",marker='o',linewidth=2)
    plt.plot(M_2, 2*h, label="$O(h)$",color='green')
    plt.plot(M_2, 71*h**2, label="$O(h^2)$",color='yellow')
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel("Number of points M")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("loglogtask1dAMR.pdf")
    plt.show()

plot_AMR_errors()
