import numpy as np
import matplotlib.pyplot as plt
from gll_utils import gll_pts_wts
from sem_utils import construct_c_matrix, construct_m_matrix, construct_b_vector

pi = np.pi



def forward_euler_step(M_inv,C,f_vec,dt,u_0):
    cfl = np.max(u_0)*dt/(2/len(f_vec))
    if(cfl>0.9):
        print(f'Stability warning: CFL = {cfl}')

    u_1 = u_0 + dt*M_inv@(f_vec-C@u_0)

    return u_1

def restrict_periodic_matrices(M,C):
    '''
    Reduces the system to enforce first-last node periodic boundary conditions
    '''
    M[:,0] += M[:,-1]
    M[0,:] += M[-1,:]
    C[:,0] += C[:,-1]
    C[0,:] += C[-1,:]

    return M[:-1,:-1],C[:-1,:-1]

def restrict_periodic_vector(f):
    f[0] += f[-1]
    return f[:-1]

def solve_1d_advection(f,N,u0,dt,t_f):
    (x_pts, _) = gll_pts_wts(N)
    times = np.arange(dt,t_f+dt,dt)
    u0_vec = u0(x_pts) # initial condition
    U = np.zeros((len(times),N))
    U[0,:] = u0_vec[:-1]

    M = construct_m_matrix(N)
    C = construct_c_matrix(N,int(1.5*N)+1)
    M,C = restrict_periodic_matrices(M,C)
    M_inv = np.linalg.inv(M)

    for i, t in enumerate(times[:-1]):
        if(i%100==0):
            print(f'Solving timestep {i}. Max u: {np.max(U[i,:])}')
        f_n = lambda x: f(x,t) # Forcing function at timestep t_n
        f_vec = construct_b_vector(N,f_n)
        f_vec = restrict_periodic_vector(f_vec)
        U[i+1,:] = forward_euler_step(M_inv,C,f_vec,dt,U[i,:])
    
    return U

def plot_advection_solution(U, x_pts, times, exact_fn=None, t_idx=None):
    """
    Plots the numerical and (optionally) exact solution at a given time index.
    """
    if t_idx is None:
        t_idx = len(times) // 2  # Middle time step by default
    plt.figure(figsize=(8, 5))
    plt.plot(x_pts[:-1], U[t_idx], 'r--', label='Numerical')
    if exact_fn:
        plt.plot(x_pts[:-1], exact_fn(x_pts[:-1], times[t_idx]), 'b', label='Exact')
    plt.title(f"t = {times[t_idx]:.2f}")
    plt.legend(); plt.grid(); plt.show()

if __name__ == '__main__':
    # Parameters
    N, tf, dt = 15, 2.0, 0.002

    # Domain and initial condition
    (x_pts, _) = gll_pts_wts(N)
    c = 1.0  # Advection speed

    # Sine wave pulse initial condition
    u0 = lambda x: np.sin(2 * np.pi * x)

    # Forcing function (zero for pure advection)
    f = lambda x, t: 0.0

    # Exact solution for comparison
    exact_sol = lambda x, t: np.sin(2 * np.pi * ((x - c * t) % 1))

    # Solve
    times = np.arange(dt, tf + dt, dt)
    U = solve_1d_advection(f, N, u0, dt, tf)

    # Plot at several time steps
    for t_idx in [0, len(times)//4, len(times)//2, -1]:
        plot_advection_solution(U, x_pts, times, exact_fn=exact_sol, t_idx=t_idx)
