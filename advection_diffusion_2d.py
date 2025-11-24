import numpy as np
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points, construct_solution_2d_fast
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from sem_utils import construct_ax_matrix_2d, construct_ay_matrix_2d, construct_load_matrix_2d, construct_m_matrix_2d, modify_A_b_dirichlet_2D, \
    map_4d_to_2d, map_2d_to_1d, map_1d_to_2d, construct_cx_matrix_2d, construct_cy_matrix_2d, construct_cx_cy_overintegrated_fast
import sympy as sp
import time
from plotting_utils import animate_solution_2d

'''
    Uses the Spectral element method to solve:
        ∂u/∂t = −cₓ ∂u/∂x − c_y ∂u/∂y + α (∂²u/∂x² + ∂²u/∂y²) + f,   x,y ∈ [−1, 1]²

    Nomenclature:
    N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials so N+1 GLL points     
'''

pi = np.pi

def backward_euler_step(N,Ax,Ay,Cx,Cy,M,f,u_0,BCs,dt):
    '''
    Advances the solution by one time step using the backward Euler method for the 2D transient Poisson equation.

    Arguments:
        N    - polynomial order (number of GLL points per dimension is N+1)
        Ax   - stiffness matrix in the x-direction (4D)
        Ay   - stiffness matrix in the y-direction (4D)
        Cx   - advection matrix in the x-direction (4D)
        Cy   - advection matrix in the y-direction (4D)
        M    - mass matrix (4D)
        f    - load matrix at the new time step (2D array, shape (N+1, N+1))
        u_0  - solution matrix at the previous time step (2D array, shape (N+1, N+1))
        BCs  - array of 4 boundary functions, [U_L, U_R, U_B, U_T]
        dt   - time step size

    Returns:
        u1   - solution matrix at the new time step (2D array, shape (N+1, N+1))
    '''
    LHS_matrix = map_4d_to_2d(M/dt + Ax + Ay + Cx + Cy, N)
    RHS_vector = map_2d_to_1d(f+np.tensordot(M/dt, u_0, axes=([2, 3], [0, 1])),N)
    # RHS_vector = map_2d_to_1d(M@u_0/dt + f,N)
    LHS_matrix, RHS_vector = modify_A_b_dirichlet_2D(BCs, LHS_matrix, RHS_vector,N)
    u1 = np.linalg.solve(LHS_matrix, RHS_vector)
    return map_1d_to_2d(u1,N)

def bdf_ext_2(N, Ax, Ay, Cx, Cy, M, f, u_0, u_1,BCs,dt):
    LHS_matrix = map_4d_to_2d(3*M/(2*dt) + Ax + Ay,N)
    RHS_vector = map_2d_to_1d(f+\
                            np.tensordot(M/(2*dt), (4*u_1-u_0), axes=([2, 3], [0, 1])) - \
                            np.tensordot((-Cx-Cy), (2*u_1-u_0), axes=([2, 3], [0, 1])),N)
    LHS_matrix, RHS_vector = modify_A_b_dirichlet_2D(BCs, LHS_matrix, RHS_vector,N)
    u2 = np.linalg.solve(LHS_matrix,RHS_vector)
    return map_1d_to_2d(u2,N)

def bdf_ext_3(N, Ax, Ay, Cx, Cy, M, f, u_0, u_1,u_2,BCs,dt):
    LHS_matrix = map_4d_to_2d(11*M/(6*dt) + Ax + Ay,N)
    RHS_vector = map_2d_to_1d(f+\
                            np.tensordot(M/(6*dt), (18*u_2-9*u_1+2*u_0), axes=([2, 3], [0, 1])) - \
                            np.tensordot((-Cx-Cy), (3*u_2-3*u_1+u_0), axes=([2, 3], [0, 1])),N)
    LHS_matrix, RHS_vector = modify_A_b_dirichlet_2D(BCs, LHS_matrix, RHS_vector,N)
    u2 = np.linalg.solve(LHS_matrix,RHS_vector)
    return map_1d_to_2d(u2,N)

def transient_solution(f,c,alpha,N,u0,bcs,dt,t_f):
    '''
    Computes the transient solution of the 2D Poisson equation using backward Euler time stepping.

    Arguments:
        f    - forcing function, callable, f(x, y, t)
        c    - vector containing two wavespeed functions c=[c_x(x,y), c_y(x,y)]
        alpha - diffusion coefficient
        N    - polynomial order (number of GLL points per dimension is N+1)
        u0   - initial condition, f(x, y), callable, vectorized
        bcs  - list of 4 Dirichlet boundary condition functions [U_L, U_R, U_B, U_T]
        dt   - time step size
        t_f  - final time

    Returns:
        U    - array of solution arrays at each time step, shape (num_steps, N+1, N+1)
    '''
    (x_pts, _) = gll_pts_wts(N)
    times = np.arange(dt,t_f+dt,dt)
    X, Y =np.meshgrid(x_pts, x_pts) 
    u0_arr = u0(X, Y) # Initial field
    # u0_arr = u0(x_pts,x_pts) # Initial field
    U = np.zeros((len(times),N+1,N+1))
    U[0,:] = u0_arr

    # Steady Quantities computed:
    print(f'Computing Ax, Ay, Cx, Cy, and M')
    Ax = construct_ax_matrix_2d(N, alpha)
    Ay = construct_ay_matrix_2d(N, alpha)
    Cx, Cy = construct_cx_cy_overintegrated_fast(N,int(1.5*N),c)
    # Cx = construct_cx_matrix_2d(N,c[0])
    # Cy = construct_cy_matrix_2d(N,c[1])

    M = construct_m_matrix_2d(N)
    print(f'Complete, beginning timestepping')
    
    # # Backward Euler 
    # for i, t in enumerate(times[:-1]):
    #     if((i+1)%100==0):
    #         print(f'Computing step i={i} at t={t}')
    #     f_N = lambda x,y: f(x,y,t)
    #     f_arr = construct_load_matrix_2d(N, f_N) # evaluated at next timestep which is t (specific to backward euler)
    #     U[i+1,:,:] = backward_euler_step(N,Ax,Ay,Cx,Cy,M,f_arr,U[i,:,:],bcs,dt)

    # BDF2-EXT2
    # U[1,:] = u0_arr
    # for i, t in enumerate(times[:-2]):
    #     if((i+1)%100==0):
    #         print(f'BDF/EXT2 Computing step i={i} at t={t}')
    #     f_N = lambda x,y: f(x,y,t)
    #     f_arr = construct_load_matrix_2d(N, f_N) # evaluated at next timestep which is t (specific to backward euler)
    #     U[i+2,:,:] = bdf_ext_2(N,Ax,Ay,Cx,Cy,M,f_arr,U[i,:,:],U[i+1,:,:],bcs,dt)

    # BDF3-EXT3
    U[1,:] = u0_arr
    U[2,:] = u0_arr
    for i, t in enumerate(times[:-3]):
        if((i+1)%100==0):
            print(f'BDF/EXT3 Computing step i={i} at t={t}')
        f_N = lambda x,y: f(x,y,t)
        f_arr = construct_load_matrix_2d(N, f_N)
        U[i+3,:,:] = bdf_ext_3(N, Ax, Ay, Cx, Cy, M, f_arr, U[i,:,:], U[i+1,:,:], U[i+2,:,:], bcs, dt)

    return U, times

if __name__ == '__main__':
    N=8
    (gll_pts, gll_wts) = gll_pts_wts(N)
    tf = 5.5
    dt = 0.006

    alpha = 0.0001
    u_0 = lambda x, y: np.exp(-((x-0.25)**2 + (y-0.25)**2) / (2*0.1**2))  # Updated initial condition
    f = lambda x,y,t:0
    bc = lambda xy: 0.0 # Homogenous dirichlet BCs for now
    bcs = [bc,bc,bc,bc]
    cx = lambda x,y: -1*y
    cy = lambda x,y: 1*x
    c = [cx, cy]

    U_solution, times = transient_solution(f, c, alpha, N, u_0, bcs, dt, tf)

    # # Plot the solution at the final time
    # plot_solution(U_solution, times, gll_pts, tf)

    # Create and save an animation of the solution
    animate_solution_2d(U_solution, times, gll_pts,'./Figures/advection_diffusion_transient_solution.mp4')





