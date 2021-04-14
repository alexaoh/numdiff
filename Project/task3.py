"""Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.

h and k, i.e. step sizes in x and y respectively, DO NOT have to be equal. 
"""
from scipy.sparse import diags # Make sparse matrices with scipy.
import scipy.sparse.linalg
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from plotting_utilities import plot3d_sol
from utilities import *

class Task3:
          
    def plot_solution(self, Mx, My):
        """Calculate numerical solution on specified grid, and plot together with analytical solution."""
        U, xv, yv = self.num_solution_Mx_My(Mx = 50, My = 50)
        plot3d_sol(U, xv, yv, Uan = self.analytic_solution)

    def convergence_plot(self, varying = None, power1 = 1.5, power2 = 2.0, savename = False):
        """Make convergence plot specified by which quantity is varying.
        
        FORKLAR MER OM HVILKE ARGUMENTER VARYING TAR!
        """
        if varying == "Mx":
            self._varying = "Mx"
            self._constant_list = [20, 50, 100, 500]
            maximum = 2**11
        elif varying == "My":
            self._varying = "My"
            self._constant_list = [20, 50, 100, 500]
            maximum = 2**11
        elif varying == "Both":
            self._varying = "Both"
            maximum = 2**10

        self._power1 = power1
        self._power2 = power2

        varying_list = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
        
        if self._varying == "Both":
            self._discrete_error = np.zeros(len(varying_list))
            for i in range(len(varying_list)):
                Usol, xv, yv = self.num_solution_Mx_My(Mx = varying_list[i], My = varying_list[i])
                analsol = self.analytic_solution(xv, yv)
                self._discrete_error[i] = e_l(Usol, analsol)
            if savename:
                assert(isinstance(savename, str))
                self.plot_plots(varying_list, varying_list, savename=savename)
            else: 
                self.plot_plots(varying_list, varying_list)
        elif self._varying:
            for constant in self._constant_list:
                self._discrete_error = np.zeros(len(varying_list))
                for i, m in enumerate(varying_list):
                    if self._varying == "Mx":
                        Usol, xv, yv = self.num_solution_Mx_My(Mx = m, My = constant)
                    elif self._varying == "My":
                        Usol, xv, yv = self.num_solution_Mx_My(Mx = constant, My = m)

                    analsol = self.analytic_solution(xv, yv)
                    self._discrete_error[i] = e_l(Usol, analsol)

        self.plot_plots()

    def analytic_solution(self, x, y):
        """Analytical solution to the 2D Laplace equation."""
        return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

    def num_solution_Mx_My(self, Mx, My):
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
                            np.full(M2, 1/h**2)[:-My], np.full(M2, 1/h**2)[:-My]])
        diag = np.array([-1, 0, 1, -My, My])
        A = diags(data, diag, format = "csc")
        
        # Construct F.
        F = np.zeros(M2)
        point_counter = 1

        # Change some elements to match the correct linear system + construct F. 
        for i in range(My, My*Mx, My):
            A[i-1, i] = A[i, i-1] = 0
            F[i-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)
            point_counter += 1

        # Add values out of loop-bound.  
        F[M2-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)

        
        # Solve linear system.  
        Usol = scipy.sparse.linalg.spsolve(A, F) 
        
        # Next, want to unpack into grids, for plotting.
        x = np.linspace(0, 1, Mx+2)
        y = np.linspace(0, 1, My+2) 

        xv, yv = np.meshgrid(x, y)
    
        U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

        # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
        U[-1, :] = np.sin(2*np.pi*x)

        # Need to unpack the solution vector with the correct coordinates. 
        for i in range(int(len(Usol)/My)): # This gives the rows (x-values).
            for j in range(My): # This gives the columns (y-values).
                U[j+1, i+1] = Usol[j + (My*i)]

        return U, xv, yv

    def plot_plots(self, Mx, My, savename = False):
            """Helper function used when plotting convergence plots."""
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(Mx*My, self._discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
            plot_order(Mx*My, self._discrete_error[0], self._power1, r"$\mathcal{O}$($h^{%s}$)" % str(self._power1), "red")
            plot_order(Mx*My, self._discrete_error[0], self._power2, r"$\mathcal{O}$($h^{%s}$)" % str(self._power2), "green")
            ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
            ax.set_xlabel(r"$M_x \cdot M_y$")
            plt.legend()
            plt.grid() 
            if savename:
                plt.savefig(savename+".pdf")
            plt.show() 
    
    
    """    
    def constant_Mx_convergence_plot(Mx, savename = False):
        maximum = 2**11
        My = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
        discrete_error = np.zeros(len(My))
        
        for i, m in enumerate(My):
            
            Usol, xv, yv = num_solution_Mx_My(Mx = Mx, My = m)
            analsol = analytic_solution(xv, yv)

            discrete_error[i] = e_l(Usol, analsol)

        def plot_plots(savename = savename):
            power = 1.5
            power2 = 2.0
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(Mx*My, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
            plot_order(Mx*My, discrete_error[0], power, r"$\mathcal{O}$($h^{%s}$)" % str(power), "red")
            plot_order(Mx*My, discrete_error[0], power2, r"$\mathcal{O}$($h^{%s}$)" % str(power2), "green")
            ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
            ax.set_xlabel(r"$M_y \cdot M_y$")
            fig.suptitle(r"$M_x = $"+str(Mx)+" constant")
            plt.legend()
            plt.grid() 
            if savename:
                plt.savefig(savename+".pdf")
            plt.show() 

        plot_plots()

    def convergence_plot_both_varying(savename=False):
        maximum = 2**10
        My = Mx = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
        discrete_error = np.zeros(len(My))
        
        for i in range(len(My)):
            Usol, xv, yv = num_solution_Mx_My(Mx = Mx[i], My = My[i])
            analsol = analytic_solution(xv, yv)
            discrete_error[i] = e_l(Usol, analsol)

        def plot_plots(savename = savename):
            power = 1.0
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(Mx*My, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
            plot_order(Mx*My, discrete_error[0], power, r"$\mathcal{O}$($h^{%s}$)" % str(power), "red")
            ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
            ax.set_xlabel(r"$M_y \cdot M_y$")
            fig.suptitle(r"$M_x$"+" and "+r"$M_y$"+" varying")
            plt.legend()
            plt.grid() 
            if savename:
                plt.savefig(savename+".pdf")
            plt.show() 

        plot_plots()



#convergence_plot_both_varying()
"""

both = Task3()
both.convergence_plot("Both", power1=1.0, power2=1.5)
