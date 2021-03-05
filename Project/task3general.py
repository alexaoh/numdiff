"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

Trying to generalize the methods used in task3.py, in order to plot convergence plots eventually. 
"""

from task3 import *
from task1a import *

def num_solution_uniform_grid(M):
    """Numerical solution of 2D Laplace on a uniform grid. Takes M = number of internal points in x and y."""

    # Construct A. 
    data = np.array([np.full(M**2, 1), np.full(M**2, -4), np.full(M**2, 1), 
                        np.full(M**2, 1), np.full(M**2, 1)])
    diags = np.array([-1, 0, 1, -M, M])
    A = spdiags(data, diags, M**2, M**2).toarray()

    # Construct F.
    F = np.zeros(M**2)
    point_counter = 1

    # List for upper boundary conditions. 
    #upper_bc = np.zeros(M+2)
    # This is done in a better way below IMO. 
    # Could perhaps also be improved for F later. 

    # Change some elements to match the correct linear system + construct F. 
    for i in range(M, M*M, M):
        A[i-1, i] = A[i, i-1] = 0
        F[i-1] = -np.sin(2*np.pi*point_counter/(M+1))
        #upper_bc[point_counter] =  np.sin(2*np.pi*point_counter/(M+1))
        point_counter += 1
    
    # Add values out of loop-bound.  
    #upper_bc[point_counter] = np.sin(2*np.pi*point_counter/(M+1))
    F[M**2-1] = -np.sin(2*np.pi*point_counter/(M+1))

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)
    
    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    #U[:, -1] = upper_bc
    t = np.linspace(0, 1, M+2)
    U[:, -1] = np.sin(2*np.pi*t)

    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/M)): # This gives the rows (x-values).
        for j in range(1,M+1): # This gives the columns (y-values).
            U[j, i+1] = Usol[i+(M*(j-1))]
            
    return U, xv, yv

def anal_solution(x, y):
    """Analytic solution to the 2D Laplace equation."""
    return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

#U, xv, yv = num_solution_uniform_grid(M = 9)
#plot3d_sol(U,xv, yv,  Uan=anal_solution)


## Make convergence plots below. Dette må lages spesifikt for hver retning tenker jeg. 

M = np.arange(3, 1012/5, 100, dtype = int)
discrete_error = np.zeros(len(M))
cont_error = np.zeros(len(M))

for i, m in enumerate(M):
    x = np.linspace(0, 1, m+2)
    Usol, xv, yv = num_solution_uniform_grid(m)
    analsol = anal_solution(xv, yv)

    discrete_error[i] = e_l(Usol, analsol)

    #interpU = interp1d(x, Usol, kind = 'cubic')
    #cont_error[i] = e_L(interpU, anal_solution, x[0], x[-1]) # Denne e_L må nok lages spesifikt til denne oppgaven!

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(M, discrete_error, label=r"$e^r_l$", color = "blue", linewidth = 3)
#ax.plot(M, cont_error, label = r"$e^r_{L_2}$", color = "yellow", linestyle = "--", linewidth = 2)
#ax.plot(M, (lambda x: 1/x**2)(M), label=r"$O$($h^2$)", color = "red", linewidth = 2)
ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
ax.set_xlabel("Number of points M")
plt.legend()
plt.grid() 
#plt.savefig("convergencePlotTask3.pdf")
plt.show() 
