import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly
from gll_utils import gll_pts_wts, integrate_gll, print_matrix

'''
    Uses the Spectral element method to solve:
        d²u/dx² = f(x),   x ∈ [−1, 1]

    Nomenclature:
    N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials        
'''
pi = np.pi

# Problem Constants: 
x_a, x_b = -1, 1 # x_a<=x<=x_
forcing_function = lambda x: pi**2/4 * np.cos(pi*x/2)
u_L, u_R = 0.0, 0.0

def construct_a_matrix(N):
    A = np.zeros((N+1,N+1))
    (gll_pts,_) = gll_pts_wts(N)
    for i in range(N+1):
        L_i_prime = create_lagrange_derivative(i, gll_pts)
        L_i = create_lagrange_poly(i, gll_pts)
        for j in range(N+1):
            L_j_prime = create_lagrange_derivative(j, gll_pts)
            Lij_prime = lambda x: L_i_prime(x)*L_j_prime(x)
            A[i,j] = integrate_gll(x_a, x_b, Lij_prime, N) - L_j_prime(1)*L_i(1) + L_j_prime(-1)*L_i(-1)
    return A

def construct_b_vector(N,f):
    b = np.zeros(N+1)
    (gll_pts,_) = gll_pts_wts(N)
    for i in range(N+1):
        L_i = create_lagrange_poly(i,gll_pts)
        f_L_i = lambda x: L_i(x)*f(x)
        b[i] = integrate_gll(x_a, x_b, f_L_i, N)

    return b

def modify_A_b_dirichlet(A,b,u_L,u_R):
    # Uses row replacement to enforce dirichlet boundary conditions
    A[0,:]=0.0
    A[0,0]=1.0
    A[-1,:]=0.0
    A[-1,-1]=1.0
    b[0]=u_L
    b[-1]=u_R
    return A,b

if __name__ == '__main__':
    N=9
    A = construct_a_matrix(N)
    print('A matrix:')
    print_matrix(A)
    b = construct_b_vector(N, forcing_function)
    print('B vector:')
    print_matrix(b)

    # modify A and b with row replacement
    A,b = modify_A_b_dirichlet(A,b,0.0,0.0)

    u = np.linalg.solve(A, b)
    print('U(solution) vector:')
    print_matrix(u)

    (pts,_) = gll_pts_wts(N)
    print('solution points:')
    print_matrix(pts)