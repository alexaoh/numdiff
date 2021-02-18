from scipy.sparse import spdiags # Make sparse matrices with scipy.
import numpy as np

def crank_nicolson(V, x, t, M, N, g0 = 0, g1 = 0):
    """Crank-Nicolson's method.

    V: Grid.
    M: Number of internal points in x-dimension in the grid. 
    N: Number of internal points in t-direction in the grid.  
    g0: Left boundary condition (1D)
    g1: Right boundary condition (1D)
    
    Returns a new list X with the solution of the problem in the gridpoints. 
    """
    X = V.copy()
    
    # Calculate step lengths. 
    h = 1/(M+1)
    k = t[1]-t[0]

    # Set up the array Ah with scipy.sparse.spdiags, 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M).toarray()*1/h**2
    
    for n in range(N):
        # For generality, even though g0 and g1 are zero here. 
        X[0, n+1] = g0(t[n+1])
        X[M+1, n+1] = g1(t[n+1])
        
        # Make temporary matrix and vector to make code more verbose. 
        B = np.identity(M)-k/2*Ah
        b = (np.identity(M)+k/2*Ah)@X[1:-1,n]

        # Solve problem Bl = b with np.linalg.solve. 
        l = np.linalg.solve(B, b)

        # Assign x to Y[:, n+1]
        X[1:-1, n+1] = l
        
        # Could do the two last steps in the same step. 
    return X

    # Do not think we will use the function above, but will leave it here for now, just in case. 

def trapezoidal_method(V, Q, t, h): # Could/should probably be further generalized later. 
    """Solves \dot{V} = \frac{1}{h^2}QV on time axis t.
    
    V: Grid to solve on.
    Q: Matrix in right hand side of equation to solve. 
    t: Time axis.
    N: Steps in time.
    h: Step length in x.
    """

    sol = np.zeros((len(t),len(V)))
    sol[0,:] = V
    k = t[1]-t[0]
    M = Q.shape[0]
    r = k/h**2
    print(Q.shape)
    print(sol.shape)
    for n in range(len(t)-1):
        lhs = (np.eye(M) - (r/2)*Q)
        rhs = (np.eye(M) + (r/2)*Q) @ sol[n, :]
        l = np.linalg.solve(lhs, rhs)
        sol[n+1, :] = l
    return sol
    