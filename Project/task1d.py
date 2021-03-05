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
# problems in first order func;
# 1) error flattens out
# 2) looks more like second order to me...


def coeff_stencil(i,h): # i=1 to i=M
    #d_p, d_m, d_2m = d_i+1, d_i-1, d_i-2
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
    """Makes the matrix Ah, the discretization of U_xx, dependent on the grid x."""
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    h = np.zeros(len(x)-1)
    h[:] = x[1:] - x[:-1]
    
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,h) 
    #care for the indices; a[0] = a_1 in the scheme
    
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


def num_sol_AMR_first(x):  #uses the 3-point stencil with non-uniform grid
    M = len(x)-2
    h = np.zeros(len(x)-1)
    h[:] = x[1:] - x[:-1]
 
    b = 2/(h[:-1]*(h[:-1]+h[1:]))    #b[0] = b_1, b[-1] = b_M, b is M long
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


def AMR(x0, steps, num_solver): 
    """Uses mesh refinement 'steps' times. Finds the error, x-grid and numerical solution for each step."""
    disc_error = np.zeros(steps+1)
    M_list = np.zeros(steps+1)
    Usol_M = [num_solver(x0)]
    X_M = [x0]
    for k in range(steps):
        M_list[k] = len(X_M[-1])-2
        
        u = anal_solution(X_M[-1])        
        disc_error[k] = la.norm(Usol_M[-1] - u)/la.norm(u) #relative disc_error
        
        error_ave = np.average(np.abs(Usol_M[-1]-u))
        
        x = np.copy(X_M[-1])
        
        #refinement, cell-solution (works for 2nd order)
        #share the x-axis into intervals and split the interval if average error at boundaries are higher than error_bound
        n = 0
        error_list = (np.abs(Usol_M[-1][:-1]-u[:-1]) + np.abs(Usol_M[-1][1:]-u[1:]))/2
        for i in range(len(error_list)):
            if error_list[i] > error_ave:
                x = np.insert(x,i+n+1,(X_M[-1][i]+X_M[-1][i+1])/2)        
                n += 1
        
        if (x[2]-x[1]) != (x[1]-x[0]):  #ensure first and second interval are equally long, this might not be necessary for first order
            x = np.insert(x, 1, x[1]/2)
            
        '''
        ## Here, tried something new; Compare x-axis to the x-axis two steps earlier, if an interval has not yet been refined -> refine it.
        if k>=2:
            h_now = x[1:] - x[:-1]
            h_earlier = X_M[k-2][1:] - X_M[k-2][:-1]
            if np.isclose(np.amax(h_now),np.amax(h_earlier)):
                indices = np.where(np.isclose(h_now,np.amax(h_now)))[0]
                z = np.copy(x)
                
                for j in range(len(indices)):
                    i = indices[j]
                    x = np.insert(x,i+1+j,(z[i]+z[i+1])/2)
          
        '''
         
        '''
        #back + forward refinement
        for i in range(1,len(Usol_M[-1])-1): #know the correct values at the boundary
            if abs(Usol_M[-1][i]-u[i]) > error_ave:
                x = np.insert(x,i+n+1,X_M[-1][i] + (X_M[-1][i+1]-X_M[-1][i])/4)     
                x = np.insert(x,i+n,X_M[-1][i]-(X_M[-1][i]-X_M[-1][i-1])/4)
                n += 2
        if (x[2]-x[1]) != (x[1]-x[0]):  #insert a node to ensure first and second interval are equally long
            x = np.insert(x, 1, x[1]/2)
        '''
        
        '''
        #----Jostein sin implementasjon
        # Forward refinement
        #Function for forward refinement, ensures first and second interval are equally long.
        n = 0
        i = 2
        while (i < len(x)-1):
            if abs(Usol_M[-1][i-n]-u[i-n]) > 0.5*error_ave:
                x = np.insert(x, i+1, (x[i]+x[i+1])/2)
                i += 2
                n += 1
            else:
                i += 1
        if abs(Usol_M[-1][1]-u[1]) > error_ave:
            x = np.insert(x, 2, (x[1]+x[2])/2)
            x = np.insert(x, 1, (x[0]+x[1])/2)
        '''
        '''
        #Dette var også et forsøk på å diskretisere i siste endepunkt, men sannsynligvis ikke hensiktsmessig likevel. 
        #    if abs(Usol_M[-1][-2]-u[-2]) > error_ave:
        #     x = np.append(x, (X_M[-1][-3]+X_M[-1][-2])/2)


        ##### Alex tried something else below, but the order flattens out and the error does not decrease (only flattens for second order however!!). 
        x = list(np.copy(X_M[-1]))
        
        for i in range(len(Usol_M[-1])):
            if abs(Usol_M[-1][i]-u[i]) > error_ave:  #It works with 0.5*error_ave, perhaps becuase then it is almost like UMR.
                small_list = [x[i], (x[i]+x[i+1])/2] 
                x[i] = small_list
            else:
                x[i] = [x[i]]
                
        x = np.array(sum(x, []))
        '''
        X_M.append(x)
        Usol_M.append(num_solver(x))
        
    #add last elements
    u = anal_solution(X_M[-1])
    disc_error[-1] = la.norm(Usol_M[-1]-u)/la.norm(u)
    M_list[-1] = len(X_M[-1])-2
    return Usol_M, X_M, disc_error, M_list


# testing the first or second order AMR solver
def test_plot_AMR_solver(num_solver):
    """Encapsulation for ease of use under development."""
    M = 9
    x = np.linspace(0, 1, M+2)
    steps = 4
    U, X, _, _ = AMR(x,steps,num_solver)
    for i in range(0,steps+1):
        #print(X[i])
        plt.plot(X[i],U[i],label=str(i))

    plt.plot(X[-1],anal_solution(X[-1]),label="An",linestyle='dotted')
    plt.legend()
    plt.show()

#test_plot_AMR_solver(num_sol_AMR_second)

#plotting error, here disc_error. Also need cont_error!
M = 9
x0 = np.linspace(0, 1, M+2)
steps = 10 # =16

U_1, X_1, disc_error_1, M_1 = AMR(x0,steps,num_sol_AMR_first)
U_2, X_2, disc_error_2, M_2 = AMR(x0,steps,num_sol_AMR_second)
 
#for i in range(steps+1):
#    print(X_2[i][:10])
# Does not behave quite like in the presentations

def plot_bar_error(X,U,start,stop):
    if stop > steps + 1:
        return 0
    
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(15,15))
    for i in range(start,stop):    
        h = X[i][1:] - X[i][:-1]
        u = anal_solution(X[i])
        ave = np.average(np.abs(U[i]-u))
        x = (X[i][1:]+X[i][:-1])/2
        error = (np.abs(U[i][1:]-u[1:]) + np.abs(U[i][:-1]-u[:-1]))/2
        axs.flatten()[i].plot(X[i],np.ones_like(X[i])*ave,label='ave',linestyle='dashed')
        axs.flatten()[i].bar(x, error, width=h,align='center',label=str(i))
    
    #plt.legend()
    fig.tight_layout()
    plt.show()

#plot_bar_error(X_1,U_1,0,steps)

def plot_AMR_errors():
    h = 1/(M_2+1)
    plt.plot(M_1, disc_error_1, label="e_l_first")
    plt.plot(M_2, disc_error_2, label="e_l_second")
    plt.plot(M_2, 80*h, label="O(h)", linestyle='dashed')
    plt.plot(M_2, 80*h**2, label="O(h^2)", linestyle='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()

plot_AMR_errors()
