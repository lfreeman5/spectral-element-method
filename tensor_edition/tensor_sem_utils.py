import numpy as np
from gll_utils import gll_pts_wts
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly

def create_Bhat_Dhat(N):    
    pts, wts = gll_pts_wts(N)
    Bhat = np.diag(wts)
    lagrange_derivs = [create_lagrange_derivative(i, pts) for i in range(N+1)]
    ld_vals = np.array([[lagrange_derivs[i](pts[j])for j in range(N+1)] for i in range(N+1)])
    Dhat = ld_vals.T
    return Bhat, Dhat

def create_mass_stiffness_2d(N):
    Bhat, Dhat = create_Bhat_Dhat(N)
    D = Dhat.T@Bhat@Dhat
    M = np.kron(Bhat, Bhat)
    A = np.kron(D,Bhat) + np.kron(Bhat,D) # Note that this isn't scaled by alpha
    return M,A     

def create_C(N,M,CxM,CyM):
    pass

def map_2d_to_1d(arr, N):
    '''
    Maps a NxN array to N^2x1 column vector
    Equivalent to U = u.reshape(-1,order='F')
    '''
    oneD = np.zeros((N+1)*(N+1))
    for i in range(N+1):
        for j in range(N+1):
            k=i+(N+1)*j
            oneD[k] = arr[i,j]
    return oneD

def map_1d_to_2d(vec, N):
    '''
    Maps an N^2x1 column vector to NxN array
    Equivalent to U = u.reshape((N+1, N+1), order='F')
    '''
    twoD = np.zeros((N+1,N+1))
    for k in range((N+1)*(N+1)):
        i = np.mod(k,N+1)
        j = (k-i)//(N+1)
        twoD[i,j] = vec[k]
    return twoD

def modify_lhs_rhs_dirichlet(LHS,RHS,N,u_dirichlet):
    pass
    # Left and right
    for j in range(N+1):
        kL = (N+1)*j
        kR = (N+1)*j + N
        LHS[kL,:] = 0.
        LHS[kR,:] = 0.
        LHS[kL, kL] = 1.
        LHS[kR, kR] = 1.
        RHS[kL] = u_dirichlet
        RHS[kR] = u_dirichlet
    # Top and bottom
    for i in range(N+1):
        kB = i
        kT = i+(N+1)*N
        LHS[kB,:] = 0.
        LHS[kT,:] = 0.
        LHS[kB, kB] = 1.
        LHS[kT, kT] = 1.
        RHS[kB] = u_dirichlet
        RHS[kT] = u_dirichlet

    return LHS,RHS