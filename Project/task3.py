"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

Trying to generalize the methods used in task3.py, in order to plot convergence plots eventually. 
"""

from scipy.sparse import diags # Make sparse matrices with scipy.
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
from plotting_utilities import plot3d_sol
from task3SpecialCases import analytic_solution

def num_solution_Mx_My(Mx, My):
    """Numerical solution of 2D Laplace.
    
    Input: 
    Mx: Number of internal points in x-direction. 
    My: Number of internal points in y-direction. 
    """

    M2 = Mx*My

    # Construct A. 
    data = np.array([np.full(M2, 1)[:-1], np.full(M2, -4), np.full(M2, 1)[:-1], 
                        np.full(M2, 1)[:-My], np.full(M2, 1)[:-My]]) # Litt jalla måte å gjøre det på, men måtte fikse rett lengde på diagonalene. 
    diag = np.array([-1, 0, 1, -My, My])

    A = diags(data, diag, format = "csc")
    
    # Construct F.
    F = np.zeros(M2)
    point_counter = 1

    # List for upper boundary conditions. 
    #upper_bc = np.zeros(M+2)
    # This is done in a better way below IMO. 
    # Could perhaps also be improved for F later. 
    # I have not removed this yet, just in case I find something wrong with the other method. 

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
    # Changed to a sparse solver in order to solve larger systems. If not enough; change to an iterative solver. 
    Usol = scipy.sparse.linalg.spsolve(A, F) 
    
    # Next, want to plot the solution. 

    x = np.linspace(0, 1, Mx+2)
    y = np.linspace(0, 1, My+2) 

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    #U[:, -1] = upper_bc
    U[-1, :] = np.sin(2*np.pi*x)

    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/My)): # This gives the rows (x-values).
        for j in range(My): # This gives the columns (y-values).
            U[j+1, i+1] = Usol[j + (My*i)]

    return U, xv, yv

U, xv, yv = num_solution_Mx_My(Mx = 100, My = 100)
plot3d_sol(U, xv, yv)
