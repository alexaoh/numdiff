"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

This file contains the analytical solution of the 2D Laplace equation in task 3. 

It also contains three special cases of the numerical solution:
1) Mx = My = 3
2) Mx = My = 9
3) Mx = My = M (numerical solution for generic M on uniform grid)
"""

from scipy.sparse import spdiags # Make sparse matrices with scipy.
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
from plotting_utilities import plot3d_sol

def analytic_solution(x, y):
    """Analytical solution to the 2D Laplace equation."""
    return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

def num_solution_M3():
    """Numerical solution of the 2D Laplace with five point stencil, with M_x = M_y = 3 =: M.
    
    This solution matches the example from the report. 
    """

    M = 3

    # Construct A. 
    data = np.array([np.full(M**2, 1), np.full(M**2, -4), np.full(M**2, 1), 
                        np.full(M**2, 1), np.full(M**2, 1)])
    diags = np.array([-1, 0, 1, -3, 3])
    A = spdiags(data, diags, M**2, M**2).toarray()

    # Change some elements to match the correct linear system. 
    A[3, 2], A[2, 3], A[6, 5], A[5, 6] = 0, 0, 0, 0

    # Construct F.
    F = np.zeros(M**2)
    F[2] = -np.sin(2*np.pi*1/4)
    F[5] = -np.sin(2*np.pi*2/4)
    F[8] = -np.sin(2*np.pi*3/4)

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)

    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    U[-1, :] = np.array([0, np.sin(2*np.pi*1/4), np.sin(2*np.pi*2/4), np.sin(2*np.pi*3/4), 0]) 
    
    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/3)):
        U[1, i+1] = Usol[i]
        U[2, i+1] = Usol[i+3]
        U[3, i+1] = Usol[i+6]


    return U, xv, yv

def num_solution_M9():
    """Numerical solution of the 2D Laplace with five point stencil, with M_x = M_y = 9 =: M.
    
    This method has 9*9 = 81 unknown internal grid points.  
    """

    M = 9

    # Construct A. 
    data = np.array([np.full(M**2, 1), np.full(M**2, -4), np.full(M**2, 1), 
                        np.full(M**2, 1), np.full(M**2, 1)])
    diags = np.array([-1, 0, 1, -9, 9])
    A = spdiags(data, diags, M**2, M**2).toarray()

    # Change some elements to match the correct linear system. 
    A[9, 8], A[8, 9], A[18, 17], A[17, 18] = 0, 0, 0, 0
    A[27, 26], A[26, 27], A[36, 35], A[35, 36] = 0, 0, 0, 0
    A[45, 44], A[44, 45], A[54, 53], A[53, 54] = 0, 0, 0, 0
    A[63, 62], A[62, 63], A[72, 71], A[71, 72] = 0, 0, 0, 0

    # Construct F.
    F = np.zeros(M**2)
    F[8] = -np.sin(2*np.pi*1/10)
    F[17] = -np.sin(2*np.pi*2/10)
    F[26] = -np.sin(2*np.pi*3/10)
    F[35] = -np.sin(2*np.pi*4/10)
    F[44] = -np.sin(2*np.pi*5/10)
    F[53] = -np.sin(2*np.pi*6/10)
    F[62] = -np.sin(2*np.pi*7/10)
    F[71] = -np.sin(2*np.pi*8/10)
    F[80] = -np.sin(2*np.pi*9/10)

    # Solve linear system. 
    Usol = la.solve(A, F) # Think about whether or not we should use a sparse solver instead (this is also true for the other tasks.)
    
    # Next, want to plot the solution. 

    x = y = np.linspace(0, 1, M+2) # Gives three internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    U[-1, :] = np.array([0, np.sin(2*np.pi*1/10), np.sin(2*np.pi*2/10), np.sin(2*np.pi*3/10), 
            np.sin(2*np.pi*4/10), np.sin(2*np.pi*5/10), np.sin(2*np.pi*6/10),  np.sin(2*np.pi*7/10), 
                np.sin(2*np.pi*8/10), np.sin(2*np.pi*9/10), 0]) 
    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/9)):
        U[1, i+1] = Usol[i]
        U[2, i+1] = Usol[i+9]
        U[3, i+1] = Usol[i+18]
        U[4, i+1] = Usol[i+27]
        U[5, i+1] = Usol[i+36]
        U[6, i+1] = Usol[i+45]
        U[7, i+1] = Usol[i+54]
        U[8, i+1] = Usol[i+63]
        U[9, i+1] = Usol[i+72]

    return U, xv, yv

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
    # I have not removed this yet, just in case I find something wrong with the other method. 

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

    x = y = np.linspace(0, 1, M+2) # Gives internal points + boundaries.

    xv, yv = np.meshgrid(x, y)

    U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

    # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
    #U[:, -1] = upper_bc
    U[-1, :] = np.sin(2*np.pi*x) 

    # Need to unpack the solution vector with the correct coordinates. 
    for i in range(int(len(Usol)/M)): # This gives the rows (x-values).
        for j in range(M): # This gives the columns (y-values).
            U[j+1, i+1] = Usol[j + (M*i)]

    return U, xv, yv
