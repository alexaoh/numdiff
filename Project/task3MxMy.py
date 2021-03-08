"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

Trying to generalize the methods used in task3.py, in order to plot convergence plots eventually. 
"""

from task3 import *
from task1a import *
from task3general import num_solution_uniform_grid

def num_solution_Mx_My(Mx, My):
    """Numerical solution of 2D Laplace.
    
    Input: 
    Mx: Number of internal points in x-direction. 
    My: Number of internal points in y-direction. 
    """

    M2 = Mx*My

    # Construct A. 
    data = np.array([np.full(M2, 1), np.full(M2, -4), np.full(M2, 1), 
                        np.full(M2, 1), np.full(M2, 1)])
    diags = np.array([-1, 0, 1, -My, My])

    A = spdiags(data, diags, M2, M2).toarray()
    
    # Construct F.
    F = np.zeros(M2)
    point_counter = 1

    # List for upper boundary conditions. 
    #upper_bc = np.zeros(M+2)
    # This is done in a better way below IMO. 
    # Could perhaps also be improved for F later. 

    # Change some elements to match the correct linear system + construct F. 
    for i in range(My, My*Mx, My):
        A[i-1, i] = A[i, i-1] = 0
        F[i-1] = -np.sin(2*np.pi*point_counter/(Mx+1))
        #upper_bc[point_counter] =  np.sin(2*np.pi*point_counter/(M+1))
        point_counter += 1
    
    # Add values out of loop-bound.  
    #upper_bc[point_counter] = np.sin(2*np.pi*point_counter/(M+1))
    F[M2-1] = -np.sin(2*np.pi*point_counter/(Mx+1))
    
    # Solve linear system. 
    Usol = la.solve(A, F) # The solver needs to be changed in order to solve larger systems!
    
    # Next, want to plot the solution. 

    x = np.linspace(0, 1, Mx+2)
    y = np.linspace(0, 1, My+2) 

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    
    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    #U[:, -1] = upper_bc
    #t = np.linspace(0, 1, Mx+2)
    U[-1, :] = np.sin(2*np.pi*x)

    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/My)): # This gives the rows (x-values).
        j = 0
        while j < My: # This gives the columns (y-values).
            U[j+1, i+1] = Usol[j+(My*i)]
            j += 1
            
    return U, xv, yv

def analytic_solution(x, y):
    """Analytic solution to the 2D Laplace equation."""
    return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

U, xv, yv = num_solution_Mx_My(Mx = 150, My = 20)
plot3d_sol(U, xv, yv)
