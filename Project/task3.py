"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

k \\neq h, i.e. step sizes in both directions DO NOT have to be equal. 
"""

from scipy.sparse import diags # Make sparse matrices with scipy.
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from plotting_utilities import plot3d_sol

def analytic_solution(x, y):
    """Analytical solution to the 2D Laplace equation."""
    return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

def num_solution_Mx_My(Mx, My):
    """Numerical solution of 2D Laplace.
    
    Input: 
    Mx: Number of internal points in x-direction. 
    My: Number of internal points in y-direction. 
    """

    M2 = Mx*My

    h = 1/(Mx+1)
    k = 1/(My+1)

    # Construct A. 
    data = np.array([np.full(M2, 1/k**2)[:-1], np.full(M2, -2*(1/h**2+1/k**2)), np.full(M2, 1/k**2)[:-1], 
                        np.full(M2, 1/h**2)[:-My], np.full(M2, 1/h**2)[:-My]]) # Litt jalla måte å gjøre det på, men måtte fikse rett lengde på diagonalene. 
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
        F[i-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)
        #upper_bc[point_counter] =  np.sin(2*np.pi*point_counter/(M+1))
        point_counter += 1

    # Add values out of loop-bound.  
    #upper_bc[point_counter] = np.sin(2*np.pi*point_counter/(M+1))
    F[M2-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)

    
    # Solve linear system. 
    # Changed to a sparse solver in order to solve larger systems. If not enough; change to an iterative solver. 
    Usol = scipy.sparse.linalg.spsolve(A, F) 
    
    # Next, want to unpack into grids, for plotting later. 
    
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

U, xv, yv = num_solution_Mx_My(Mx = 50, My = 50)
plot3d_sol(U, xv, yv, Uan = analytic_solution)

# Make convergence plot (for discontinuous norm). 
# Generalize later!

def constant_My_convergence_plot(My):
    """Convergence plot with My kept constant and Mx increasing."""
    maximum = 2**11
    Mx = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
    discrete_error = np.zeros(len(Mx))
    for i, m in enumerate(Mx):
        Usol, xv, yv = num_solution_Mx_My(Mx = m, My = My)
        analsol = analytic_solution(xv, yv)

        discrete_error[i] = la.norm(analsol-Usol)/la.norm(analsol)

    def plot_plots(savename = False):
        """Encapsulation for development purposes."""
        power = 1.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(Mx*My, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
        ax.plot(Mx*My, (lambda x: 1/x**power)(Mx), label=r"$O$($h^{%s}$)" % str(power), color = "red", linewidth = 2)
        ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
        ax.set_xlabel(r"$M_x \cdot M_y$")
        fig.suptitle(r"$M_y = $"+str(My)+" constant")
        plt.legend()
        plt.grid() 
        if savename:
            plt.savefig(savename+".pdf")
        plt.show() 

    plot_plots()

def constant_Mx_convergence_plot(Mx):
    """Convergence plot with Mx kept constant and My increasing."""
    maximum = 2**11
    My = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
    discrete_error = np.zeros(len(My))
    
    for i, m in enumerate(My):
        
        Usol, xv, yv = num_solution_Mx_My(Mx = Mx, My = m)
        analsol = analytic_solution(xv, yv)

        discrete_error[i] = la.norm(analsol-Usol)/la.norm(analsol)

    def plot_plots(savename = False):
        """Encapsulation for development purposes."""
        power = 1.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(Mx*My, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
        ax.plot(Mx*My, (lambda x: 1/x**power)(My), label=r"$O$($h^{%s}$)" % str(power), color = "red", linewidth = 2)
        ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
        ax.set_xlabel(r"$M_y \cdot M_y$")
        fig.suptitle(r"$M_x = $"+str(Mx)+" constant")
        plt.legend()
        plt.grid() 
        if savename:
            plt.savefig(savename+".pdf")
        plt.show() 

    plot_plots()

def convergence_plot_both_varying():
    """Convergence plot with both Mx and My varying."""
    maximum = 2**10
    My = Mx = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
    discrete_error = np.zeros(len(My))
    
    for i in range(len(My)):
        
        Usol, xv, yv = num_solution_Mx_My(Mx = Mx[i], My = My[i])
        analsol = analytic_solution(xv, yv)

        discrete_error[i] = la.norm(analsol-Usol)/la.norm(analsol)

    def plot_plots(savename = False):
        """Encapsulation for development purposes."""
        power = 1.5
        power2 = 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(Mx*My, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
        ax.plot(Mx*My, (lambda x: 1/x**power)(Mx), label=r"$O$($h^{%s}$)" % str(power), color = "red", linewidth = 2)
        ax.plot(Mx*My, (lambda x: 1/x**power2)(Mx), label=r"$O$($h^{%s}$)" % str(power2), color = "green", linewidth = 2)
        ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
        ax.set_xlabel(r"$M_y \cdot M_y$")
        fig.suptitle(r"$M_x$"+" and "+r"$M_y$"+" varying")
        plt.legend()
        plt.grid() 
        if savename:
            plt.savefig(savename+".pdf")
        plt.show() 

    plot_plots("task3bBothVary")
