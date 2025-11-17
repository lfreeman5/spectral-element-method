import numpy as np
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from numpy.polynomial.legendre import leggauss, legder, legval

'''
    A set of utilities related to Operators
    Used to solve multiple problems; kept here for brevity
'''

def construct_a_matrix(N): # Only assembled once, so maybe bad performance isn't the end of the world
    # Restricted to [-1.,1.]
    A = np.zeros((N+1, N+1))
    (gll_pts, _) = gll_pts_wts(N)
    lagrange_polys = [create_lagrange_poly(i, gll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative_gll_points(i, gll_pts) for i in range(N+1)]
    for i in range(N+1):
        L_i_prime = lagrange_derivs[i]
        L_i = lagrange_polys[i]
        for j in range(N+1):
            if((j!=0 and j!=N) and (i!=0 and i!=N)): # Can't assume outside rows/cols are symmetric with non-homogenous Dirichlet BCs 
                if(A[j,i]!=0.0):
                    A[i,j]=A[j,i]
                    continue
            L_j_prime = lagrange_derivs[j]
            Lij_prime = lambda x: L_i_prime(x) * L_j_prime(x)
            A[i, j] = integrate_gll(-1, 1, Lij_prime, N) - L_j_prime(1) * L_i(1) + L_j_prime(-1) * L_i(-1)
    return A

def construct_c_matrix(N,M): # Anti-aliasing using M>N quadrature 
    # I'm not sure that I have i,j right here
    C = np.zeros((N+1,N+1)) 
    (gll_pts, _) = gll_pts_wts(N)
    lagrange_polys = [create_lagrange_poly(i, gll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative(i, gll_pts) for i in range(N+1)]
    for i in range(N+1):
        print(f'Creating row {i+1} of C')
        L_i = lagrange_polys[i]
        for j in range(N+1):
            L_j_prime = lagrange_derivs[j]
            Lij_prime = lambda x: L_i(x) * L_j_prime(x)
            C[i, j] = integrate_gll(-1, 1, Lij_prime, M)
    return C

def construct_b_vector(N,f): # Restricted to x \in [-1,1]
    b = np.zeros(N+1)
    (gll_pts,_) = gll_pts_wts(N)
    for i in range(N+1):
        L_i = create_lagrange_poly(i,gll_pts)
        f_L_i = lambda x: L_i(x)*f(x)
        b[i] = integrate_gll(-1, 1, f_L_i, N)
    return b

def construct_m_matrix(N):
    (_, gll_wts) = gll_pts_wts(N)
    M = np.diag(gll_wts)
    return M

def modify_A_b_dirichlet(A,b,u_L,u_R): # Used generally on assembled systems.
    # Uses row replacement to enforce dirichlet boundary conditions
    A[0,:]=0.0
    A[0,0]=1.0
    A[-1,:]=0.0
    A[-1,-1]=1.0
    b[0]=u_L
    b[-1]=u_R
    return A,b

def construct_ax_matrix_2d(N, alpha=1):
    '''
        Computes the diffusion operator tensor which is 4D!
        Uses the GLL points of P_N --> N+1 points --> size(A_x) = (N+1)^4 
        
        Using 4 loops probably isn't optimal... but it makes sense that way.
        Plus, cost of Ax construction is amortized. And it only takes 8s to form 64x64x64x64 matrix
    '''
    A_x = np.zeros((N+1,N+1,N+1,N+1))
    (gll_pts, gll_wts) = gll_pts_wts(N)
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    ld_vals = [[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)] # Evaluation of l_i'(xi_j)
    ld_vals = np.array(ld_vals)

    for j in range(N+1):
        for q in range(N+1):
            if(q!=j): continue
            for i in range(N+1):
                for p in range(N+1):
                    A_x[i,j,p,q] = gll_wts[j]*np.sum(ld_vals[i,:]*ld_vals[p,:]*gll_wts)

    return alpha*A_x

def construct_ay_matrix_2d(N, alpha=1):
    '''
    Similar to construct_ax_matrix_2d but creates the y-diffusion component
    '''
    A_y = np.zeros((N+1,N+1,N+1,N+1))
    (gll_pts, gll_wts) = gll_pts_wts(N)
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    ld_vals = [[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)] # Evaluation of l_i'(ξ_j)
    ld_vals = np.array(ld_vals)

    for i in range(N+1):
        for p in range(N+1):
            if(p!=i): continue
            for j in range(N+1):
                for q in range(N+1):
                    A_y[i,j,p,q] = gll_wts[i]*np.sum(ld_vals[j,:]*ld_vals[q,:]*gll_wts)

    return alpha*A_y

def construct_cx_matrix_2d(N,cx):
    '''
    creates the x-advection tensor.
    Inputs:
    N: Polynomial order
    cx: vector field of x-advection speed, cx=cx(x,y) defined on [-1,1]^2
    NOTE: This doesn't use over-integration for dealiasing. Problem? 
    '''
    C_x = np.zeros((N+1,N+1,N+1,N+1))
    (gll_pts, gll_wts) = gll_pts_wts(N)
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    ld_vals = np.array([[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)]) # Evaluation of l_i'(ξ_j) as ld_vals[i,j]
    for j in range(N+1):
        for q in range(N+1):
            if (q!=j): continue # equivalent to delta_jq. Can get rid of a loop by using q=j but it makes more sense to read this way
            for i in range(N+1):
                for p in range(N+1):
                    C_x[i,j,p,q] = gll_wts[j]*gll_wts[i] * ld_vals[p,i] * cx(gll_pts[i], gll_pts[j])

    return C_x

def construct_cx_kron_2d(N,cx):
    '''
    creates the x-advection tensor using kroneker products.
    Inputs:
    N: Polynomial order
    cx: vector field of x-advection speed, cx=cx(x,y) defined on [-1,1]^2

    NOTE: This does use over integration for dealiasing
    '''
    # greate GL points
    m = int(2.5*N)+1
    # m = N+1

    x_gl, w_gl = leggauss(m)
    (gll_pts, gll_wts) = gll_pts_wts(N)

    # create B_hat_m matrix eq. 476
    B_hat_m = np.diag(w_gl)

    # create B_m matrix eq. 479
    B_m = np.kron(B_hat_m, B_hat_m)

    # create Cx_m  page 95 of notes
    Cx_m_diag = np.zeros(m**2)
    for k in range(m):
        for l in range(m):
            i = k + m*l  # check this
            Cx_m_diag[i] = cx(x_gl[k],x_gl[l])
    # print(Cx_m_diag)
    Cx_m = np.diag(Cx_m_diag)

    # make lagrange polys and d_lagrange polys 
    lagrange_polys = [create_lagrange_poly(i, gll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    # ld_vals = np.array([[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)]) # Evaluation of l_i'(ξ_j) as ld_vals[i,j]
    
    # create J_hat eq. 477
    J_hat = np.zeros((m,N+1))
    for q in range(N+1):
        lagrange_q = lagrange_polys[q]
        for l in range(m):
            J_hat[l][q] = lagrange_q(x_gl[l])

    # create D_tilde eq. 478
    D_tilde = np.zeros((m,N+1))

    for p in range(N+1):
        d_lagrange_p = lagrange_derivs[p]
        for k in range(m):
            D_tilde[k][p] = d_lagrange_p(x_gl[k])

    # print(Cx_m)
    # print(B_m)
    # print(J_hat)
    # print(D_tilde)
    # eq 
    C_x = np.kron(J_hat.transpose(),J_hat.transpose())@B_m@Cx_m@np.kron(J_hat,D_tilde)
    # print(C_x)
    return C_x  

def construct_cy_matrix_2d(N,cy):
    '''
    y-advection tensor.
    cy = cy(x,y) on [-1,1]^2
    '''
    C_y = np.zeros((N+1,N+1,N+1,N+1))
    (gll_pts, gll_wts) = gll_pts_wts(N)
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    ld_vals = np.array([[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)]) # Evaluation of l_i'(ξ_j) as ld_vals[i,j]
    for i in range(N+1):
        for p in range(N+1):
            if(p!=i): continue # equivalent to applying delta_ip
            for j in range(N+1):
                for q in range(N+1):
                    C_y[i,j,p,q] = gll_wts[i]*gll_wts[j]*ld_vals[q,j]*cy(gll_pts[i],gll_pts[j])
    return C_y

def construct_cx_cy_overintegrated(N, M, c):
    '''
    Overintegrated advection tensor construction.
    Vectorized for speed using numpy broadcasting and sum.
    '''
    [cx, cy] = c
    C_xijpq, C_yijpq = np.zeros((N+1,N+1,N+1,N+1)), np.zeros((N+1,N+1,N+1,N+1))
    (ngll_pts, ngll_wts) = gll_pts_wts(N)
    (mgll_pts, mgll_wts) = gll_pts_wts(M)
    lagrange_polys = [create_lagrange_poly(i, ngll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative(i, ngll_pts) for i in range(N+1)]
    l_vals = np.array([[lagrange_polys[i](mgll_pts[j]) for j in range(M+1)] for i in range(N+1)])  # (N+1, M+1)
    ld_vals = np.array([[lagrange_derivs[i](mgll_pts[j]) for j in range(M+1)] for i in range(N+1)])  # (N+1, M+1)
    cx_vals = np.array([[cx(mgll_pts[n], mgll_pts[m]) for n in range(M+1)] for m in range(M+1)])  # (M+1, M+1)
    cy_vals = np.array([[cy(mgll_pts[n], mgll_pts[m]) for n in range(M+1)] for m in range(M+1)])  # (M+1, M+1)

    # Vectorized computation for all i, j, p, q
    for i in range(N+1):
        for j in range(N+1):
            for p in range(N+1):
                for q in range(N+1):
                    sx = np.sum(mgll_wts * cx_vals[:, :] * ld_vals[p, :] * l_vals[i, :], axis=1)  # shape (M+1,)
                    xval = np.sum(sx * mgll_wts * l_vals[j, :] * l_vals[q, :])
                    sy = np.sum(mgll_wts * cy_vals[:, :] * l_vals[p, :] * l_vals[i, :], axis=1)  # shape (M+1,)
                    yval = np.sum(sy * mgll_wts * l_vals[j, :] * ld_vals[q, :])
                    C_xijpq[i, j, p, q] = xval
                    C_yijpq[i, j, p, q] = yval
    return C_xijpq, C_yijpq


def construct_cx_cy_overintegrated_fast(N, M, c):
    """
    Optimized overintegrated advection tensor construction.
    """
    cx, cy = c
    ngll_pts, ngll_wts = map(np.asarray, gll_pts_wts(N))
    mgll_pts, mgll_wts = map(np.asarray, gll_pts_wts(M))

    lagrange_polys = [create_lagrange_poly(i, ngll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative(i, ngll_pts) for i in range(N+1)]

    # Basis and derivative evaluations
    l_vals = np.array([[lagrange_polys[i](mgll_pts[j]) for j in range(M+1)] for i in range(N+1)])
    ld_vals = np.array([[lagrange_derivs[i](mgll_pts[j]) for j in range(M+1)] for i in range(N+1)])

    # Coefficient fields (must support array inputs)
    X, Y = np.meshgrid(mgll_pts, mgll_pts, indexing='ij')
    cx_vals = cx(X, Y)
    cy_vals = cy(X, Y)

    # Weighted coefficients
    Wx = np.outer(mgll_wts, mgll_wts)
    Cx = cx_vals * Wx
    Cy = cy_vals * Wx

    # Final contractions
    C_xijpq = np.einsum('mn,pm,im,qn,jn->ijpq', Cx, ld_vals, l_vals, l_vals, l_vals)
    C_yijpq = np.einsum('mn,pm,im,qn,jn->ijpq', Cy, l_vals, l_vals, ld_vals, l_vals)

    return C_xijpq, C_yijpq

def construct_m_matrix_2d(N):
    '''
        Computes the mass matrix in 2D: size(M) = (N+1)^4
        It's diagonal.
    '''
    (_, gll_wts) = gll_pts_wts(N)
    M = np.zeros((N+1,N+1,N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            M[i,j,i,j] = gll_wts[i]*gll_wts[j]
    return M

def construct_load_matrix_2d(N,func):
    '''
        Computes f_ij, the load matrix (vector). Size(f)=(N+1)^2

        f must be callable of the form f(x,y) ie it is valid for a single timestep (not f(x,y,t))
    '''
    f = np.zeros((N+1,N+1))
    (gll_pts,gll_wts) = gll_pts_wts(N)
    for i in range(N+1):
        for j in range(N+1):
            f[i,j] = func(gll_pts[i],gll_pts[j])*gll_wts[i]*gll_wts[j]
    return f

def map_4d_to_2d(A,N):
    '''
    maps a_ijpq to a_kl
    Assumes square A (ie size of all 4 dimensions is equal to N+1)
    '''
    B = np.zeros(((N+1)*(N+1),(N+1)*(N+1)))
    for i in range(N+1):
        for j in range(N+1):
            for p in range(N+1):
                for q in range(N+1):
                    k = i+j*(N+1)
                    l = p+q*(N+1)
                    B[k,l]=A[i,j,p,q]
    return B

def map_2d_to_1d(A,N):
    '''
    maps a_ij to a_k
    Assumes square a_ij, both sides of dimension N+1
    '''
    F = np.zeros(((N+1)*(N+1)))
    for i in range(N+1):
        for j in range(N+1):
            k = i+j*(N+1)
            F[k]=A[i,j]
    return F

def map_1d_to_2d(F,N):
    '''
    reverse of map_2d_to_1d
    maps F_k of size (N+1)^2 to a_ij (N+1)x(N+1)
    '''
    A = np.zeros((N+1,N+1))
    for k in range((N+1)*(N+1)):
        i = np.mod(k,N+1)
        j = np.floor_divide(k,N+1)
        A[i,j] = F[k]
    return A

def modify_A_b_dirichlet_2D(bcs,A,b,N):
    """
    Applies Dirichlet boundary conditions to the flattened 2D system matrix A and vector b.

    Parameters:
        bcs (list): List of four functions [u_L, u_R, u_B, u_T] specifying boundary values on the left, right, bottom, and top edges, respectively.
        A (ndarray): Flattened 2D system matrix of shape ((N+1)**2, (N+1)**2).
        b (ndarray): Flattened right-hand side vector of length (N+1)**2.
        N (int): Polynomial order (number of GLL points is N+1).

    Returns:
        tuple: Modified (A, b) with Dirichlet boundary conditions enforced.
    """
    [u_L, u_R, u_B, u_T] = bcs
    (pts, _) = gll_pts_wts(N)
    # LHS i=0
    for j in range(N+1):
        y_j = pts[j]
        k=j*(N+1)
        b[k] = u_L(y_j)
        A[k,:] = 0.0
        A[k,k] = 1.0    
    # RHS i=N
    for j in range(N+1):
        y_j = pts[j]
        k=N+j*(N+1)
        b[k] = u_R(y_j)
        A[k,:] = 0.0
        A[k,k] = 1.0
    # Bottom j=0:
    for i in range(N+1):
        x_i = pts[i]
        k = i
        b[k] = u_B(x_i)
        A[k,:] = 0.0
        A[k,k] = 1.0
    # Top j=N:
    for i in range(N+1):
        x_i = pts[i]
        k = i+N*(N+1)
        b[k] = u_T(x_i)
        A[k,:] = 0.0
        A[k,k] = 1.0

    return A, b
