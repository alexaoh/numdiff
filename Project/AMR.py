''' AMR code for 2b has been moved here'''

## ----- AMR -----
from scipy.sparse import diags # Make sparse matrices with scipy.

def first_order_AMR(x,t): #not sure if this stencil is right w.r.t t-axis, just tried something.

    M = len(x)-2
    N = len(t)-1
    U = np.zeros((N+1,M+2))
    U[0,:] = initial(x)
    U[:,0] = 0
    U[:,-1] = 0
    
    h = x[1:] - x[:-1]
    k = t[1:] - t[:-1]
    
    b = 2/(h[:-1]*(h[:-1]+h[1:]))    
    c = 2/(h[1:]*(h[1:] + h[:-1]))

    data = [b[1:], -(b+c), c[:-1]]   #how does the B.C fit in here?
    diagonals = np.array([-1, 0, 1])
    Ah = diags(data, diagonals).toarray()

    for n in range(N):  #Backward Euler
        b = U[n,1:-1]
        lhs = np.identity(M)-k[n]*Ah
        U[n+1,1:-1] = la.solve(lhs,b)
    return U

def calc_cell_errors(U,u):
    '''Calulculates an error for each cell by taking the avg of the error in the endpoints.'''
    n = len(u) - 1 # Number of cells
    cell_errors = np.zeros(n)
    for i in range(n):
        cell_errors[i] = abs(U[i] - u[i]) #(np.abs(u[i] - U[i]) + np.abs(u[i + 1] - U[i + 1]))
    return cell_errors

def AMR(x0,t0,steps,type):   #type = t,h-refinement, maybe r also
    disc_error = np.zeros(steps+1)
    Usol_M = [first_order_AMR(x0,t0)]
    refine_M = []
    ref_list = np.zeros(steps+1)
    
    if type == 'h':
        refine_M = [x0]
        for k in range(steps):
            ref_list[k] = len(refine_M[-1])-2

            #x_1,t_1 = np.meshgrid(refine_M[-1],t0)
            u = anal_solution(refine_M[-1],t0[-1]) #look at the error in last iteration
            disc_error[k] = e_l(Usol_M[-1][-1,:],u)  

            x = list(np.copy(refine_M[-1]))
            u = anal_solution(refine_M[-1],t0[1])
            cell_errors = calc_cell_errors(Usol_M[-1][1,:],u) #look at the first iteration in t to minmialize error in t when we refine h.
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
            refine_M.append(x)
            Usol_M.append(first_order_AMR(x,t0))
            
        u = anal_solution(refine_M[-1],t0[-1]) 
        disc_error[-1] = e_l(Usol_M[-1][-1,:],u)
        ref_list[-1] = len(refine_M[-1])-2
        
    if type == 't':
        refine_M = [t0]
        for k in range(steps):
            ref_list[k] = len(refine_M[-1])-1

            x_1,t_1 = np.meshgrid(x0,refine_M[-1])
            u = anal_solution(x_1,t_1) 
            disc_error[k] = e_l(Usol_M[-1][-1,:],u[-1,:])   #look at the error in last iteration

            t = list(np.copy(refine_M[-1]))
            u_ave_t = np.average(u,axis=1)
            U_ave_t = np.average(Usol_M[-1],axis=1)
            cell_errors = calc_cell_errors(U_ave_t,u_ave_t) #use avereage error along x for each t as a cell, this might be wrong!
            tol = 1 * np.average(cell_errors)

            j = 0 # Index for x in case we insert points

            # For testing if first or second cell have been refined
            #firstCell = False
            for i in range(len(cell_errors)):
                if cell_errors[i] > tol:
                    t.insert(j+1, t[j] + 0.5 * (t[j+1] - t[j]))
                    j += 1
                    '''
                    # Tests to ensure that first and second cell have same length
                    if i == 0:
                        firstCell = True
                        x.insert(j+1, x[j] + 0.5 * (x[j+1] - x[j]))
                        j += 1
                    if i == 2 and not firstCell:
                        x.insert(1, x[0] + 0.5 * (x[1] - x[0]))
                        j += 1
                    '''
                j += 1
            t = np.array(t)
            refine_M.append(t)
            Usol_M.append(first_order_AMR(x0,t))
            
        u = anal_solution(x0,t0[-1]) 
        disc_error[-1] = e_l(Usol_M[-1][-1,:],u)
        ref_list[-1] = len(refine_M[-1])-1
    
    return Usol_M, refine_M, disc_error, ref_list


t_max = 0.2

# h-refinement    
N = 1000
t = np.linspace(0,t_max,N+1)
M0 = 9
x0 = np.linspace(0,1,M0+2)

steps = 13
U , X, disc_error, M = AMR(x0,t,steps,'h')

'''
for i in range(steps):
    plt.plot(X[i],U[i][-1,:])
plt.plot(X[-1],anal_solution(X[-1],T),label='an',linestyle='dashed')
plt.legend()
plt.show()
    
'''
plt.plot(M*N, disc_error)
plt.xscale('log')
plt.yscale('log')
plt.show()

'''
# t-refinement
M = 1000
x = np.linspace(0,1,M+2)
N0 = 9
t0 = np.linspace(0,t_max,N0+1)

steps = 8
U , T, disc_error, N = AMR(x,t0,steps,'t')

for i in range(3,steps):
    plt.plot(x,U[i][-1,:])
plt.plot(x,anal_solution(x,t_max),label='an',linestyle='dashed')
plt.legend()
plt.show()

plt.plot(M*N, disc_error)
plt.xscale('log')
plt.yscale('log')
plt.show()
'''

'''
#not sure how to determine the error for x- and t-axis. Suggestion is below. 
#x-refinement
error_ave_x = np.average(np.abs(Usol_M[-1]-u),axis=0)
e_x = (error_ave_x[1:] + error_ave_x[:-1])/2
error_ave = np.average(error_ave_x)

x = np.copy(X_M[-1])
n = 0
for i in range(len(e_x)):
    if e_x[i] > error_ave:
        x = np.insert(x,i+n+1,(X_M[-1][i]+X_M[-1][i+1])/2)        
        n += 1

#t-refinement
error_ave_t = np.average(np.abs(Usol_M[-1]-u),axis=1)
e_t = (error_ave_t[1:] + error_ave_t[:-1])/2
error_ave = np.average(error_ave_t)

t = np.copy(T_M[-1])
n = 0
for i in range(len(e_t)):
    if e_t[i] > error_ave:
        t = np.insert(t,i+n+1,(T_M[-1][i]+T_M[-1][i+1])/2)        
        n += 1
''' 