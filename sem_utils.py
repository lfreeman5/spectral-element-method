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