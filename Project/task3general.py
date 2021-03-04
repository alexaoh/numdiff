"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

Trying to generalize the methods used in task3.py, in order to plot convergence plots eventually. 
"""

from task3 import *

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
    upper_bc = np.zeros(M+2)

    # Change some elements to match the correct linear system + construct F. 
    for i in range(M, M*M, M):
        A[i-1, i] = A[i, i-1] = 0
        F[i-1] = -np.sin(2*np.pi*point_counter/(M+1))
        upper_bc[point_counter] =  np.sin(2*np.pi*point_counter/(M+1))
        point_counter += 1
    
    # Add values out of loop-bound.  
    upper_bc[point_counter] = np.sin(2*np.pi*point_counter/(M+1))
    F[M**2-1] = -np.sin(2*np.pi*point_counter/(M+1))

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)
    
    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    U[-1, :] = upper_bc
    
    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/M)): # This gives the rows (x-values).
        
        for j in range(1,M+1): # This gives the columns (y-values).
            U[j, i+1] = Usol[i+(M*(j-1))]
            
    return U, xv, yv


U, xv, yv = num_solution_uniform_grid(M = 50)
plot3d_sol(U, xv, yv)
