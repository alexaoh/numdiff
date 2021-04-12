"""Random utility functions are placed here for now."""

import numpy as np

def forward_subsSOR(LU,b):
    ''' Forover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun L trengs i denne rutinen
        b Vektor med høyresiden i problemet som skal løses
    Output:
        u Løsningen av det lineære nedretriangulære systemet Lc=b
    '''
    n, m = LU.shape
    u = np.zeros(n)
    u[0] = b[0]/LU[0,0]
    for i in range(1,n):
        u[i] = (b[i]-LU[i,:i] @ u[:i])/LU[i,i]
        
    return u

def SOR(A, b, omega,u0,tol,maxiter):
    '''
    SOR method.
    Return the computed solution u.
    omega: Value of the relaxation parameter in SOR
    u0: The initial value for the iteration
    tol: The tolerance to be used in the stopping criterion (est < tol)
    maxiter: The maximum number of iterations
    '''
    k = 0
    est = 2*tol
    L = np.tril(A,-1)
    D = np.diag(np.diag(A))
    U = np.triu(A,1)
    dividend = omega*L+D
    dividend_inv = np.linalg.inv(dividend) #This can be used for the solution without using forward_subs (numpy calculates inverse)
    while est > tol:
        print("Iteration ",k,u0)
        k += 1   
        x = (1-omega)*(D@u0)-(omega*(U@u0))+omega*b

        u = dividend_inv@(((1-omega)*D@u0)-(omega*(U@u0))+omega*b)

        #In case using numpy to find inverse is considered "cheating", I solved it wiht forward_substitution instead/also
        #u = forward_subsSOR(dividend,x)

        est = np.linalg.norm(u-u0)
        if k >= maxiter:
            break
        u0 = u #Moves the iteration along
    return u

def mylu(A):
    '''Returns: 
    Vector P interpreted as a Pivot matrix (represents a matrix with unit vectors e_Pk in row k).
    LU is a copy of A, where the diagonal and above is U and below the diagonal is L. 
    '''
    LU = A.astype(float) #Copies A and casts all values in A to float! (Important!)
    n = A.shape[0]
    P = np.arange(n)
    for k in range(n-1):
        pivot = np.argmax(abs(LU[P[k:], k]))+k
        P[pivot], P[k] = P[k], P[pivot]
        mults = LU[P[k+1:],k] / LU[P[k],k]
        LU[P[k+1:], k+1:] = LU[P[k+1:], k+1:] - np.outer(mults, LU[P[k],k+1:])
        LU[P[k+1:], k] = mults
    return LU, P
    

def forward_subs(LU,P,b):
    ''' Forover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun L trengs i denne rutinen
        P Permutasjonsvektor av heltall
        b Vektor med høyresiden i problemet som skal løses
    Output:
        c Løsningen av det lineære nedretriangulære systemet Lc=Pb
    '''
    n, m = LU.shape
    Pb = b[P]
    c = np.zeros(n)
    c[0] = Pb[0]
    for k in range(1,n):
        c[k] = Pb[k] - LU[P[k],0:k] @ c[0:k]
        
    return c

def backward_subs(LU,P,c):
    ''' Bakover substitusjonsalgoritme
    Input:
        LU inneholder både L og U, selv om kun U trengs i denne rutinen
        P Permutasjonsvektor av heltall
        c Vektor med høyreside, dvs rutinen løser Ux=c
    Output:
        x Løsningen av det lineære øvretriangulre problemet Ux = c
    '''
    n,m = LU.shape
    x = np.zeros(n)
    x[n-1] = c[n-1]/LU[P[n-1],n-1]
    for k in range(n-1,0,-1):
        x[k-1] = (c[k-1]-LU[P[k-1],k:] @ x[k:])/LU[P[k-1],k-1]
        
    return x

