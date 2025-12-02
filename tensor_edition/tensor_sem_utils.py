import numpy as np
from gll_utils import gll_pts_wts
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly
from numpy.polynomial.legendre import leggauss, legder, legval

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


def create_C(N,M,CxM2,CyM2):
    '''
    Creates the 2D advection operator C
    N - polynomial order of the spectral element
    M - polynomial order for overintegration
    CxM2 and CyM2 are M^2x1, either evaluating C at M-gll pts either directly or by interpolation (as would be the case for Navier Stokes)
    '''
    BhatN, DhatN = create_Bhat_Dhat(N)
    BhatM, _ = create_Bhat_Dhat(M)
    JhatM = create_Jhat(N,M)
    DtildeM = create_Dtilde(N,M)
    Cx = (np.kron(JhatM.T,JhatM.T))@(np.kron(BhatM, BhatM))@np.diag(CxM2)@(np.kron(JhatM,DtildeM))
    Cy = (np.kron(JhatM.T,JhatM.T))@(np.kron(BhatM, BhatM))@np.diag(CyM2)@(np.kron(DtildeM,JhatM))
    return Cx + Cy

def map_2d_to_1d(arr, N):
    '''
    Maps a NxN array to N^2x1 column vector
    Equivalent to U = u.reshape(-1,order='F')
    '''
    oneD = np.zeros(((N+1)*(N+1)))
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


def nonlinear_advection_at_previous_time(u_coefs, v_coefs, J_hat, B_M, D_tilde):      
    
    # notes eq 222
    Cu_field = np.kron(J_hat,J_hat)@u_coefs 
    Cv_field = np.kron(J_hat,J_hat)@v_coefs 

    
    Cu_M = np.diag(Cu_field)
    Cv_M = np.diag(Cv_field)

    # notes eq 489
    Cu = np.kron(J_hat.transpose(),J_hat.transpose())@B_M@Cu_M@np.kron(J_hat, D_tilde)
    # notes eq 490
    Cv = np.kron(J_hat.transpose(),J_hat.transpose())@B_M@Cv_M@np.kron(D_tilde,J_hat)
    # maybe a faster way in eq. 491
    
    C = Cu + Cv
    Cv_u = C@u_coefs
    Cv_v = C@v_coefs

    return Cv_u, Cv_v

# unused
def nonlinear_advection_at_previous_time_textbook(N, ux_coefs, uy_coefs):    

    J_hat_N_N = create_Jhat(N,N) 
    B_hat_N, D_hat = create_Bhat_Dhat(N)
    D_tilde_N_N = create_Dtilde(N,N)
    B_N = np.kron(B_hat_N, B_hat_N)


    JJ = np.kron(J_hat_N_N, J_hat_N_N)
    v_underbarx = np.kron(J_hat_N_N, J_hat_N_N)@ux_coefs
    v_underbary = np.kron(J_hat_N_N, J_hat_N_N)@uy_coefs
    
    
    # textbook page 174 (page 202 of pdf)
    # M_hat = np.diag(coarse_wts)
    # textbook eq 4.3.17
    # M = np.kron(M_hat, M_hat)

    # or 
    M = B_N

    # textbook eq. 2.4.9
    Dhat = D_tilde_N_N # pretty sure this D_tilde is equivalent
    eye = np.identity(N+1)

    # textbook above 6.4.9
    D1 = np.kron(eye, Dhat)
    D2 = np.kron(Dhat, eye)

    # textbook page 305 (page 333 of pdf)
    V1 = np.diag(ux_coefs)
    V2 = np.diag(uy_coefs)

    # textbook eq 6.4.10
    Cvx = M@(V1@D1 + V2@D2)@v_underbarx
    Cvy = M@(V1@D1 + V2@D2)@v_underbary



    return Cvx, Cvy


def AB_coefs(k):
    '''
    k-th order Adams Bashforth coefficients, 
    returns an array of length k
    '''
    # textbook table 3.2.1
    b_k_j = [np.asarray([1.0]), np.asarray([(3/2), -(1/2)]), np.asarray([(23/12), -(16/12), (5/12)]), np.asarray([(55/24), -(59/24), (37/24), -(9/24)])]
    
    # k-1 because of python zero based indexing
    return b_k_j[k-1]

def BDFk_coefs(k):
    '''
    k-th order BDF coefficients, 
    returns the b0 value for that k 
    and beta (k-j) array of length k
    '''

    # textbook table 3.2.3
    b_k_0 = np.asarray([1., (2/3) , (6/11) , (12/25)])
    a_k_j = [np.asarray([1]),   np.asarray([(4/3), -(1/3)]),    np.asarray([(18/11), -(9/11), (2/11)]), np.asarray([(48/25), -(36/25), (16/25), -(3/25)])]

    # textbook eq 6.2.13
    beta_k         = 1/b_k_0
    beta_k_minus_j = [-a_k_j[0]/b_k_0[0],  -a_k_j[1]/b_k_0[1],    -a_k_j[2]/b_k_0[2],     -a_k_j[3]/b_k_0[3]]

    # k-1 because of python zero based indexing
    return beta_k[k-1], beta_k_minus_j[k-1] 


