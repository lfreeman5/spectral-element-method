import numpy as np
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
from gll_utils import gll_pts_wts, integrate_gll, print_matrix


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

def construct_ax_matrix_2d(N):
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

    return A_x

def construct_ay_matrix_2d(N):
    '''
    Similar to construct_ax_matrix_2d but creates the y-diffusion component
    '''
    A_y = np.zeros((N+1,N+1,N+1,N+1))
    (gll_pts, gll_wts) = gll_pts_wts(N)
    lagrange_derivs = [create_lagrange_derivative(i,gll_pts) for i in range(N+1)]
    ld_vals = [[lagrange_derivs[i](gll_pts[j])for j in range(N+1)] for i in range(N+1)] # Evaluation of l_i'(xi_j)
    ld_vals = np.array(ld_vals)

    for i in range(N+1):
        for p in range(N+1):
            if(p!=i): continue
            for j in range(N+1):
                for q in range(N+1):
                    A_y[i,j,p,q] = gll_wts[i]*np.sum(ld_vals[j,:]*ld_vals[q,:]*gll_wts)

    return A_y

def construct_m_matrix_2d(N):
    '''
        Computes the mass matrix in 2D: size(M) = (N+1)^4
        It's diagonal.
    '''
    (_, gll_wts) = gll_pts_wts(N)
    M = np.zeros((N+1,N+1,N+1,N+1))
    for i in range(N+1):
        M[i,i,i,i] = gll_wts[i]**2.
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