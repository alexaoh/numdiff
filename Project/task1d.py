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

    #Construct Dirichlet boundary condition from manuactured solution
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

    # Add left Dirichlet condition to solution.
    Usol = np.insert(Usol, 0, alpha)

    # Add right Dirichlet condtion to solution.
    Usol = np.append(Usol,beta)
    return Usol

### Uniform mesh refinement UMR, here using disc_error
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
    
    interpU_first = interp1d(x, first_order_num, kind = 'cubic')
    interpU_second = interp1d(x, second_order_num, kind = 'cubic')

    # Continuous norms. 
    e_1_cont[i] = e_L(interpU_first, anal_solution, x[0], x[-1])
    e_2_cont[i] = e_L(interpU_second, anal_solution, x[0], x[-1])

def plot_errors_UMR(save = False):
    """Encapsulation, to avoid commenting while development."""
    plt.plot(M_list,e_1_disc,label="l2_first", color = "red")
    plt.plot(M_list,8*h_list,label=r"O$(h)$",linestyle='dashed', color = "red")
    plt.plot(M_list,10*h_list**2,label=r"O$(h^2)$",linestyle='dashed', color = "blue")
    plt.plot(M_list,e_2_disc,label="l2_second", color = "blue")
    plt.plot(M_list,e_2_cont,label="L2_second", color = "orange", linestyle = "dotted")
    plt.plot(M_list,e_1_cont,label="L2_first", color = "green", linestyle = "dotted")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("loglogtask1dUMR.pdf")
    plt.show()


##--------- AMR--------------
# works better now. 
# problems; 
# 1) Have to discard U_-1 and set h_-1=h_0
# 2) The error is increasing at first step
# 3) Too much refinement for each step, can we reduce this somehow
# 4) seems like max_norm does not work, can't see why


def coeff_stencil(i,h): #i can go from i=1 to i=M
    #d_p, d_m, d_2m = d_i+1, d_i-1, d_i-2
    if i==1:
        d_2m = 2*h[0] #vet egt. ikke h_-1, setter h_-1=h_0
    else:
        d_2m = h[i-2] + h[i-1]
    d_p = h[i]
    d_m = h[i-1]
    a = 2 * (d_p - d_m) / (d_2m*(d_2m + d_p)*(d_2m - d_m)) 
    b = 2 * (d_2m - d_p) / (d_m*(d_2m - d_m)*(d_m + d_p))
    c = 2 * (d_2m + d_m) / (d_p*(d_m + d_p)*(d_2m + d_p))
   
    return a, b, c


def num_solution_four_point_stencil(x):
    """Makes the matrix Ah, the discretizaion of U_xx, dependent on the grid x."""
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    h = np.zeros(len(x)-1)
    h[:] = x[1:] - x[:-1]
    
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,h) 
    #care for the indices; a[0] = a_1 in the scheme
    
    #I disard U_-1, don't know how to get rid of it in the difference scheme, aka I set a[0]=0 (a_1 = 0)
    a[0] = 0
    #Note; Here using sparse.diags, not sparse.spdiags!! - only for me to better control what comes in to Ah
    data = [a[2:], b[1:], -(a+b+c), c[:-1]]   
    diagonals = np.array([-2,-1, 0, 1])
    Ah = diags(data, diagonals).toarray()
    
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    f_vec = np.full(M, f(x[1:-1]))
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


def AMR(x0, steps, method='average'):  #method can be 'average' or 'max_norm'
    """Uses mesh refinement 'steps' times. Finds the error, x-grid and numerical solution for each step."""
    disc_error = np.zeros(steps+1)
    M_list = np.zeros(steps+1)
    Usol_M = [num_solution_four_point_stencil(x0)] #using regular lists so we can append arrays of different shapes
    X_M = [x0]
    for k in range(steps):
        M_list[k] = len(X_M[-1])-2
        
        u = anal_solution(X_M[-1])        
        disc_error[k] = la.norm(Usol_M[-1] - u)/la.norm(u) #relative disc_error
        
        if method=='max_norm':
            error_bound = 0.7*np.amax(np.abs(Usol_M[-1]-u)) # 0.7*max_error
        else:
            error_bound = np.average(np.abs(Usol_M[-1]-u)) # using average error, no contribution to error at boundary
            
        #refine the grid
        x = np.copy(X_M[-1])
        n = 0 #hjelpevariabel
        for i in range(1,len(Usol_M[-1])-1): #know the correct values at the boundary
            if abs(Usol_M[-1][i]-u[i]) > error_bound:
                x = np.insert(x,i+n,(X_M[-1][i]+X_M[-1][i-1])/2)
                n += 1
        
        X_M.append(x)
        Usol_M.append(num_solution_four_point_stencil(x))
    
    #need to add the last error and M-number
    u = anal_solution(X_M[-1])
    disc_error[-1] = la.norm(Usol_M[-1]-u)/la.norm(u)
    M_list[-1] = len(X_M[-1])-2
    return Usol_M, X_M, disc_error, M_list


# testing the 4-point stencil with a plot
def test_plot_4_point_stencil():
    """Encapsulation for ease of use under development."""
    M = 10
    x = np.linspace(0, 1, M+2)
    steps = 6
    U, X, _, _ = AMR(x,steps)
    for i in range(0,steps+1,2):
        plt.plot(X[i],U[i],label=str(i))

    plt.plot(X[-1],anal_solution(X[-1]),label="An",linestyle='dotted')
    plt.legend()
    plt.show() #this looks better now :-)

#plotting error, here disc_error. Also need cont_error
# seems to be a problem with using 0.7*max_norm as error bound for refinement. Perhaps we can just stick with the average AMR? 
M = 9
x0 = np.linspace(0, 1, M+2)
steps = 10

U, X, disc_error, M = AMR(x0,steps) 

'''
# make a bar-plot of error at x-points, the boundary conditions are not included (because they are zero)
# the error does not behave as in presentation given in lectures!
for i in range(steps+1):    
    h = X[i][2:] - X[i][1:-1]
    ave = np.average(np.abs(U[i]-anal_solution(X[i]))) #average AMR
    plt.plot(X[i],np.ones_like(X[i])*ave,label='ave',linestyle='dashed') #average AMR
    #max_error = np.amax(np.abs(U[i]-anal_solution(X[i]))) #max AMR
    #plt.plot(X[i],np.ones_like(X[i])*0.7*max_error,label='inf_norm',linestyle='dashed') #max AMR
    plt.bar(X[i][1:-1],np.abs(U[i][1:-1]-anal_solution(X[i][1:-1])),width=h,align='edge',label=str(i))
    plt.legend()
    plt.show()
'''

'''
h = 1/(M+1)
plt.plot(M, disc_error, label="e_l_second")
plt.plot(M, 80*h**2, label="O(h^2)", linestyle='dashed')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.show()
'''
## we also need to implement a first order AMR. How do we do that? - need to find a scheme