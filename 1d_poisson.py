import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
import sympy as sp
import time

'''
    Uses the Spectral element method to solve:
        d²u/dx² = f(x),   x ∈ [−1, 1]

    Nomenclature:
    N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials        
'''
pi = np.pi

# Problem Constants: 
x_a, x_b = -1, 1 # x_a<=x<=x_b. Assumed to be constant and x ∈ [−1, 1]. This requirement will be removed later.

def construct_a_matrix(N): # Only assembled once, so maybe bad performance isn't the end of the world
    A = np.zeros((N+1, N+1))
    (gll_pts, _) = gll_pts_wts(N)
    start_time = time.time()
    lagrange_polys = [create_lagrange_poly(i, gll_pts) for i in range(N+1)]
    lagrange_derivs = [create_lagrange_derivative(i, gll_pts) for i in range(N+1)]
    for i in range(N+1):
        L_i_prime = lagrange_derivs[i]
        L_i = lagrange_polys[i]
        for j in range(N+1):
            L_j_prime = lagrange_derivs[j]
            Lij_prime = lambda x: L_i_prime(x) * L_j_prime(x)
            A[i, j] = integrate_gll(x_a, x_b, Lij_prime, N) - L_j_prime(1) * L_i(1) + L_j_prime(-1) * L_i(-1)
        print(f"Time to compute row {i} of A: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
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

def solve_1d_poisson(f,N,u_L,u_R):
    '''
        Solves the 1D poisson equation:
            d²u/dx² = f(x),   x ∈ [−1, 1], u(-1)=u_L, u(1)=u_R
        using order-N Lagrange polynomials on GLL points
        Arguments:
            f - the forcing function. Callable.
            N - polynomial order
            u_L - LHS dirichlet BC
            u_R - RHS dirichlet BC
        Returns:
        solution - callable function of x
    '''
    A = construct_a_matrix(N)
    b = construct_b_vector(N, f)
    A, b = modify_A_b_dirichlet(A, b, u_L, u_R)
    u = np.linalg.solve(A,b)
    solution = construct_solution(u, gll_pts_wts(N)[0])
    return solution

def compare_exact_approx(exact, approx):
    x_vals = np.linspace(x_a, x_b, 200)
    exact_vals = [exact(x) for x in x_vals]
    approx_vals = [approx(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, exact_vals, label='Exact Solution', color='blue')
    plt.plot(x_vals, approx_vals, label='Approximation', linestyle='--', color='red')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x)$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    N=50
    x = sp.Symbol('x')
    u_exact = sp.sin(10*x) # Exact solution
    f = -sp.diff(u_exact,x,2) # Compute corresponding forcing function
    exact = sp.lambdify(x,u_exact,'numpy')
    forcing_func = sp.lambdify(x,f,'numpy')
    u_L, u_R = exact(x_a), exact(x_b)

    u_approx = solve_1d_poisson(forcing_func, N, u_L, u_R)
    compare_exact_approx(exact, u_approx)
