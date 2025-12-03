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
        RHS[kL] = 1
        RHS[kR] = 5
    # Top and bottom
    for i in range(N+1):
        kB = i
        kT = i+(N+1)*N
        LHS[kB,:] = 0.
        LHS[kT,:] = 0.
        LHS[kB, kB] = 1.
        LHS[kT, kT] = 1.
        RHS[kB] = 2
        RHS[kT] = 6
    return LHS,RHS


def nonlinear_advection_at_previous_time(N, M, ux_coefs, uy_coefs):      

    J_hat = create_Jhat(N,M) 
    B_hat_M, D_hat = create_Bhat_Dhat(M)
    D_tilde = create_Dtilde(N,M)
    B_M = np.kron(B_hat_M, B_hat_M)
    
    # notes eq 222
    Cx_field = np.kron(J_hat,J_hat)@ux_coefs 
    Cy_field = np.kron(J_hat,J_hat)@uy_coefs 

    
    Cx_M = np.diag(Cx_field)
    Cy_M = np.diag(Cy_field)

    # notes eq 489
    Cx = np.kron(J_hat.transpose(),J_hat.transpose())@B_M@Cx_M@np.kron(J_hat, D_tilde)
    # notes eq 490
    Cy = np.kron(J_hat.transpose(),J_hat.transpose())@B_M@Cy_M@np.kron(D_tilde,J_hat)
    # maybe a faster way in eq. 491
    
    C = Cx + Cy
    Cvx = C@ux_coefs
    Cvy = C@uy_coefs

    return Cvx, Cvy

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


def construct_neumann_rhs_2d(neumann_funcs, N, alpha, pts=None, wts=None):
    """
    Assemble Neumann RHS contributions on reference element [-1,1]^2.

    Arguments:
        neumann_funcs : dict with keys 'left','right','bottom','top'
                        values are callables q(x,y) giving the normal derivative
                        on that face, or None if not Neumann on that face.
                        q must return ∂n u (n is outward normal).
        N             : polynomial order (N -> N+1 GLL points)
        alpha         : diffusion coefficient (multiplies the Neumann term)
        pts, wts      : optional arrays of GLL pts and weights; if None they are computed
    Returns:
        b2d : ndarray shape (N+1,N+1) of contributions that should be added to the
              load vector (before mapping to 1D).
    """
    if pts is None or wts is None:
        pts, wts = gll_pts_wts(N)
    # initialize 2D RHS contribution
    b = np.zeros((N+1, N+1), dtype=float)
    qL = neumann_funcs.get('left', None)
    qR = neumann_funcs.get('right', None)
    qB = neumann_funcs.get('bottom', None)
    qT = neumann_funcs.get('top', None)
    # LEFT boundary: x = -1, outward normal n = (-1,0)
    if qL is not None:
        x = -1.0
        for j in range(N+1):
            y = pts[j]
            qval = qL(x, y)  # q must be ∂_n u (n·grad u) 
            b[0, j] += alpha * qval * wts[j]
    # RIGHT boundary: x = +1, outward normal n = (1,0)
    if qR is not None:
        x = 1.0
        for j in range(N+1):
            y = pts[j]
            qval = qR(x, y)
            b[N, j] += alpha * qval * wts[j]
    # BOTTOM boundary: y = -1, outward normal n = (0,-1)
    if qB is not None:
        y = -1.0
        for i in range(N+1):
            x = pts[i]
            qval = qB(x, y)
            b[i, 0] += alpha * qval * wts[i]
    # TOP boundary: y = +1, outward normal n = (0,1)
    if qT is not None:
        y = 1.0
        for i in range(N+1):
            x = pts[i]
            qval = qT(x, y)
            b[i, N] += alpha * qval * wts[i]
    return b

def modify_lhs_rhs_mixed(LHS, RHS, N, BCs, alpha, pts, wts, u_field=None, c=None):
    """
    Mixed Dirichlet + Neumann for diffusion and advection (stable SEM version).
    """
    # --- DIFFUSION NEUMANN ---
    neumann_funcs = {side: None for side in ['left','right','bottom','top']}
    for side in neumann_funcs:
        entry = BCs.get(side, None)
        if entry is not None and entry[0] == 'neumann':
            neumann_funcs[side] = entry[1]

    b2d = construct_neumann_rhs_2d(neumann_funcs, N, alpha, pts, wts)
    print("max b2d neumann rhs:", np.max(np.abs(b2d)))
    RHS += map_2d_to_1d(b2d, N)
    # --- ADVECTION FLUX ---
    # if u_field is not None and c is not None:
    #     adv2d = construct_advection_flux_rhs_2d(u_field, c, N, pts, wts, BCs)
    #     RHS += map_2d_to_1d(adv2d, N)
    for j in range(N+1):
        # Left boundary
        if BCs.get('left', (None,None))[0] == 'dirichlet':
            k = j*(N+1)
            val = BCs['left'][1](-1.0, pts[j])
            LHS[k,:] = 0.; LHS[k,k] = 1.
            RHS[k] = val
        # Right boundary
        if BCs.get('right', (None,None))[0] == 'dirichlet':
            k = j*(N+1)+N
            val = BCs['right'][1](1.0, pts[j])
            LHS[k,:] = 0.; LHS[k,k] = 1.
            RHS[k] = val
    for i in range(N+1):
        # Bottom boundary
        if BCs.get('bottom', (None,None))[0] == 'dirichlet':
            k = i
            val = BCs['bottom'][1](pts[i], -1.0)
            LHS[k,:] = 0.; LHS[k,k] = 1.
            RHS[k] = val
        # Top boundary
        if BCs.get('top', (None,None))[0] == 'dirichlet':
            k = i + N*(N+1)
            val = BCs['top'][1](pts[i], 1.0)
            LHS[k,:] = 0.; LHS[k,k] = 1.
            RHS[k] = val
    return LHS, RHS

def construct_advection_flux_rhs_2d(u_field, c, N, pts, wts, BCs=None):
    """
    Compute boundary integral ∫ (c·n) u ϕ dS for SEM.
    Uses Dirichlet value on inflow boundaries, interior u_field on outflow.
    Strong Dirichlet nodes are skipped.
    """
    cx, cy = c
    rhs2d = np.zeros((N+1, N+1))

    normals = {'left':(-1,0),'right':(1,0),'bottom':(0,-1),'top':(0,1)}
    face_nodes = {
        'left': lambda k: (0, k),
        'right': lambda k: (N, k),
        'bottom': lambda k: (k, 0),
        'top': lambda k: (k, N)
    }

    for face in ['left','right','bottom','top']:
        nx, ny = normals[face]

        for k in range(N+1):
            ix, iy = face_nodes[face](k)
            x, y = pts[ix], pts[iy]

            # Skip strong Dirichlet nodes (already imposed)
            if BCs is not None and BCs.get(face, (None,None))[0] == 'dirichlet':
                continue

            vn = cx(x,y)*nx + cy(x,y)*ny

            if vn < 0:  # inflow
                u_val = 0.0 
                if BCs is not None and BCs.get(face, (None,None))[0] == 'dirichlet':
                    u_val = BCs[face][1](x, y)
                rhs2d[ix, iy] += abs(vn) * u_val * wts[k]
            else:       # outflow
                rhs2d[ix, iy] += vn * u_field[ix, iy] * wts[k]

    return rhs2d
