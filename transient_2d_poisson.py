import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from sem_utils import modify_A_b_dirichlet, construct_a_matrix, construct_b_vector, construct_m_matrix
import sympy as sp
import time

'''
    Uses the Spectral element method to solve:
        du/dt = d²u/dx²+d²u/dy²+f(x,y,t),   x,y ∈ [−1, 1]²

    Nomenclature:
    N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials        
'''

pi = np.pi

def backward_euler_step(A,M,f,u_0,u_L,u_R,dt):
    '''
    Advances the solution by one time step using the backward Euler method.
    Arguments:
        A   - stiffness matrix
        M   - mass matrix
        f   - load vector at the new time step
        u_0 - solution vector at the previous time step
        u_L - left Dirichlet boundary condition
        u_R - right Dirichlet boundary condition
        dt  - time step size
    Returns:
        u1  - solution vector at the new time step
    '''
    LHS_matrix = (M/dt + A)
    RHS_vector = 1/dt*M@u_0 + f 
    LHS_matrix, RHS_vector = modify_A_b_dirichlet(LHS_matrix,RHS_vector,u_L,u_R)
    u1 = np.linalg.solve(LHS_matrix, RHS_vector)
    return u1

def transient_solution(f,N,u_L,u_R,u0,dt,t_f):
    '''
    Computes the transient solution of the 1D Poisson equation using backward Euler time stepping.
    Arguments:
        f    - forcing function, callable
        N    - polynomial order - there will be N+1 points
        u_L  - left Dirichlet boundary condition, f(t), callable
        u_R  - right Dirichlet boundary condition, f(t), callable
        u0   - initial condition, f(x), callable, vectorized
        dt   - time step size
        t_f  - final time
    Returns:
        U    - array of solution vectors at each time step
    '''
    (x_pts, _) = gll_pts_wts(N)
    times = np.arange(dt,t_f+dt,dt)
    u0_vec = u0(x_pts)
    U = np.zeros((len(times),N+1))
    U[0,:] = u0_vec

    # Steady Quantities computed:
    print(f'Computing A and M')
    A = construct_a_matrix(N)
    M = construct_m_matrix(N)
    print(f'Complete, beginning timestepping')
    for i, t in enumerate(times[:-1]):
        if((i+1)%100==0):
            print(f'Computing step i={i} at t={t}')
        f_N = lambda x: f(x,t)
        f_vec = construct_b_vector(N,f_N) # evaluated at next timestep which is t (specific to backward euler)
        U[i+1,:] = backward_euler_step(A,M,f_vec,U[i,:],u_L(t),u_R(t),dt)    

    return U

if __name__ == '__main__':
    N=15
    (gll_pts, gll_wts) = gll_pts_wts(N)
    tf = 30
    dt = 0.02

    x,t = sp.Symbol('x'), sp.Symbol('t')
    u_exact = sp.sin(t/(x+2)) # Exact solution
    f = sp.diff(u_exact,t,1)-sp.diff(u_exact,x,2) # Compute corresponding forcing function

    # Print the problem being solved
    print("Solving the transient 1D Poisson equation (diffusion equation):")
    print("    du/dt - d²u/dx² = f(x, t)")
    print(f"with exact solution: u(x, t) = {u_exact}")
    print(f"and forcing function: f(x, t) = {f}")

    exact = sp.lambdify((x,t),u_exact,'numpy')
    forcing_func = sp.lambdify((x,t),f,'numpy')
    u_0 = lambda x_val: exact(x_val, 0)
    u_L = lambda t_val: exact(-1, t_val)
    u_R = lambda t_val: exact(1, t_val)

    U_solution = transient_solution(forcing_func, N, u_L, u_R, u_0, dt, tf)


