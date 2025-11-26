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

def create_Jhat(N,M):
    Jhat = np.zeros((M+1,N+1))
    Npts, _ = gll_pts_wts(N)
    Mpts, _ = gll_pts_wts(M)
    lagrange_polys = [create_lagrange_poly(i,Npts) for i in range(N+1)] # The N+1 GLL polynomials
    for l in range(M+1):
        for q in range(N+1):
            Jhat[l,q] = lagrange_polys[q](Mpts[l])
    return Jhat

def create_Dtilde(N,M):
    Dtilde = np.zeros((M+1,N+1))
    Npts, _ = gll_pts_wts(N)
    Mpts, _ = gll_pts_wts(M)
    lagrange_derivs = [create_lagrange_derivative(i, Npts) for i in range(N+1)]
    for k in range(M+1):
        for p in range(N+1):
            Dtilde[k,p] = lagrange_derivs[p](Mpts[k])
    return Dtilde

def create_mass_stiffness_2d(N):
    Bhat, Dhat = create_Bhat_Dhat(N)
    D = Dhat.T@Bhat@Dhat
    M = np.kron(Bhat, Bhat)
    A = np.kron(D,Bhat) + np.kron(Bhat,D) # Note that this isn't scaled by alpha
    return M,A     


def create_C(N,M,CxM,CyM):
    '''
    Creates the 2D advection operator C
    N - polynomial order of the spectral element
    M - polynomial order for overintegration
    CxM and CyM are MxM, either evaluating C at M-gll pts either directly or by interpolation (as would be the case for Navier Stokes)
    '''
    BhatN, DhatN = create_Bhat_Dhat(N)
    BhatM, _ = create_Bhat_Dhat(M)
    JhatM = create_Jhat(N,M)
    DtildeM = create_Dtilde(N,M)
    CxM2 = map_MM_to_M2M2(CxM,M) # For NS, the interpolated vector may already be M2 - so just do np.diag(CxM) instead
    CyM2 = map_MM_to_M2M2(CyM,M)
    Cx = (np.kron(JhatM.T,JhatM.T))@(np.kron(BhatM, BhatM))@CxM2@(np.kron(JhatM,DtildeM))
    Cy = (np.kron(JhatM.T,JhatM.T))@(np.kron(BhatM, BhatM))@CyM2@(np.kron(DtildeM,JhatM))
    return Cx + Cy

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

def map_MM_to_M2M2(arr,M):
    CM2M2 = np.zeros(((M+1)*(M+1),(M+1)*(M+1)))
    for i in range(M+1):
        for j in range(M+1):
            k = i + (M+1)*j
            CM2M2[k,k] = arr[i,j]
    return CM2M2 # Note that CM2M2 is clearly sparse and smarter storage could be used


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