import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points, construct_solution_2d_fast
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from sem_utils import construct_ax_matrix_2d, construct_ay_matrix_2d, construct_load_matrix_2d, construct_m_matrix_2d, modify_A_b_dirichlet_2D, \
    map_4d_to_2d, map_2d_to_1d, map_1d_to_2d, construct_cx_matrix_2d, construct_cy_matrix_2d
import sympy as sp
import time
from matplotlib.animation import FuncAnimation

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

def transient_solution(f,c,N,u0,bcs,dt,t_f):
    '''
    Computes the transient solution of the 2D Poisson equation using backward Euler time stepping.

    Arguments:
        f    - forcing function, callable, f(x, y, t)
        c    - vector containing two wavespeed functions c=[c_x(x,y), c_y(x,y)]
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
    Ax = construct_ax_matrix_2d(N)
    Ay = construct_ay_matrix_2d(N)
    Cx = construct_cx_matrix_2d(N,c[0])
    Cy = construct_cy_matrix_2d(N,c[1])

    M = construct_m_matrix_2d(N)
    print(f'Complete, beginning timestepping')

    for i, t in enumerate(times[:-1]):
        if((i+1)%100==0):
            print(f'Computing step i={i} at t={t}')
        f_N = lambda x,y: f(x,y,t)
        f_arr = construct_load_matrix_2d(N, f_N) # evaluated at next timestep which is t (specific to backward euler)
        U[i+1,:,:] = backward_euler_step(N,Ax,Ay,Cx,Cy,M,f_arr,U[i,:,:],bcs,dt)

    return U, times

def plot_solution(U_solution, times, gll_pts, t, N_pts=50):
    """
    Plots the solution at a given time t as a color plot.
    """
    t_idx = np.argmin(np.abs(times - t))
    u_func = construct_solution_2d_fast(U_solution[t_idx, :, :], gll_pts)
    
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    X, Y = np.meshgrid(x, y)
    # Pass 1D arrays to u_func, then transpose for plotting if needed
    U_plot = u_func(x, y)
    if U_plot.shape != X.shape:
        U_plot = U_plot.T  # Ensure shape matches meshgrid

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, U_plot, cmap='viridis', shading='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Solution at t = {times[t_idx]:.2f}')
    fig.colorbar(c, ax=ax)
    plt.show()

def animate_solution(U_solution, times, gll_pts, N_pts=100, filename='advection_diffusion_transient_solution.mp4'):
    """
    Creates a color plot animation of the solution evolving in time.
    Optimized for performance and outputs an MP4 file.
    The colorbar (z-axis) is fixed between 0 and 1.
    Now uses higher resolution (N_pts=100) and prints progress.
    """
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    X, Y = np.meshgrid(x, y)

    # Precompute all frames with progress printout
    num_frames = len(times)
    U_frames = np.empty((num_frames, N_pts, N_pts))
    print(f"Precomputing {num_frames} frames at {N_pts}x{N_pts} resolution...")
    for i in range(num_frames):
        if (i % max(1, num_frames // 10) == 0) or (i == num_frames - 1):
            print(f"  Frame {i+1}/{num_frames} ({100*(i+1)//num_frames}%)")
        u_func = construct_solution_2d_fast(U_solution[i, :, :], gll_pts)
        U_plot = u_func(x, y)
        if U_plot.shape != X.shape:
            U_plot = U_plot.T
        U_frames[i] = U_plot

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # Higher resolution figure
    c = ax.pcolormesh(X, Y, U_frames[0], cmap='viridis', shading='auto', vmin=-1, vmax=1)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title = ax.set_title(f'Solution at t = {times[0]:.4f}')

    def update(frame):
        U_plot = U_frames[frame]
        c.set_array(U_plot.ravel())
        title.set_text(f'Solution at t = {times[frame]:.4f}')
        c.set_clim(vmin=-1, vmax=1)
        return c, title

    anim = FuncAnimation(fig, update, frames=num_frames, blit=True)
    print(f"Saving animation to {filename} (this may take a while)...")
    # Use higher bitrate for higher quality
    anim.save(filename, writer='ffmpeg', fps=15, bitrate=8000)
    print("Animation saved.")
    plt.close(fig)

if __name__ == '__main__':
    N=8
    (gll_pts, gll_wts) = gll_pts_wts(N)
    tf = 0.15
    dt = 0.0002

    u_0 = lambda x,y: np.sin(pi*x)*np.sin(pi*y)
    f = lambda x,y,t:0
    bc = lambda xy: 0.0 # Homogenous dirichlet BCs for now
    bcs = [bc,bc,bc,bc]
    cx = lambda x,y: 15
    cy = lambda x,y: 15
    c = [cx, cy]

    U_solution, times = transient_solution(f, c, N, u_0, bcs, dt, tf)

    # # Plot the solution at the final time
    # plot_solution(U_solution, times, gll_pts, tf)

    # Create and save an animation of the solution
    animate_solution(U_solution, times, gll_pts)





