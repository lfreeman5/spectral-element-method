import numpy as np
import matplotlib.pyplot as plt
from gll_utils import gll_pts_wts
from sem_utils import construct_c_matrix, construct_m_matrix, construct_b_vector
from lagrange_utils import construct_solution
from tqdm import tqdm

pi = np.pi

def forward_euler_step(M_inv, C, f_vec, dt, u_0, min_dx, c=1.0):
    """
    Single forward Euler update for semi-discrete system.
    min_dx: smallest spacing (in x) used for a CFL estimate
    c: characteristic/advection speed (used for CFL)
    """
    # CFL estimate based on characteristic speed and smallest node spacing
    cfl = abs(c) * dt / float(min_dx)
    if cfl > 0.9:
        print(f"Stability warning: CFL = {cfl:.2e} (dt={dt:.3e}, min_dx={min_dx:.3e}, c={c})")

    u_1 = u_0 + dt * M_inv @ (f_vec - C @ u_0)

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

def solve_1d_advection(f, N, u0, dt, t_f, show_progress=False, c=1.0):
    (x_pts, _) = gll_pts_wts(N)
    times = np.arange(dt, t_f + dt, dt)
    u0_vec = u0(x_pts)  # initial condition
    U = np.zeros((len(times), N))
    U[0, :] = u0_vec[:-1]

    M = construct_m_matrix(N)
    # show_progress True will display row-creation progress inside construct_c_matrix
    C = construct_c_matrix(N, int(1.5 * N) + 1, show_progress=show_progress)
    M, C = restrict_periodic_matrices(M, C)
    M_inv = np.linalg.inv(M)
    # compute smallest spacing between the retained DOFs (exclude last periodic node)
    x_internal = np.sort(x_pts)[:-1]
    min_dx = np.min(np.diff(x_internal))

    iterator = range(len(times) - 1)
    if show_progress:
        iterator = tqdm(range(len(times) - 1), desc=f"Time stepping (N={N})")

    for i in iterator:
        t = times[i]
        # f at time t
        f_n = lambda x: f(x, t)
        f_vec = construct_b_vector(N, f_n, show_progress=False)
        f_vec = restrict_periodic_vector(f_vec)
        U[i + 1, :] = forward_euler_step(M_inv, C, f_vec, dt, U[i, :], min_dx=min_dx, c=c)

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
        x_pts_fine = np.linspace(-1,1,200)
        plt.plot(x_pts_fine, exact_fn(x_pts_fine, times[t_idx]), 'b', label='Exact')
    plt.title(f"t = {times[t_idx]:.2f}")
    plt.legend(); plt.grid(); plt.show()

def animate_advection_solution(U, x_pts, times, exact_fn=None, interval=30, title_tex=None, save_name=None):
    """
    Animates the numerical and (optionally) exact solution over time.
    """
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 5))
    line_num, = ax.plot([], [], 'r--', label='Numerical')
    line_exact = None
    if exact_fn:
        line_exact, = ax.plot([], [], 'b', label='Exact')
    ax.set_xlim(x_pts[0], x_pts[-2])
    ax.set_ylim(np.min(U), np.max(U))
    if title_tex is None:
        ax.set_title("Advection Solution Animation")
    else:
        ax.set_title(title_tex)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    # Use a figure-level legend below the axes so it is visible in saved animations
    fig.subplots_adjust(bottom=0.20)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    ax.grid()

    def init():
        line_num.set_data([], [])
        if line_exact:
            line_exact.set_data([], [])
            return line_num, line_exact
        return (line_num,)

    def animate(i):
        line_num.set_data(x_pts[:-1], U[i])
        if exact_fn:
            x_pts_fine = np.linspace(-1, 1, 200)
            line_exact.set_data(x_pts_fine, exact_fn(x_pts_fine, times[i]))
            ax.set_title(f"t = {times[i]:.2f}")
            return line_num, line_exact
        ax.set_title(f"t = {times[i]:.2f}")
        return (line_num,)

    frames = len(times)
    ani = animation.FuncAnimation(
        fig, animate, frames=frames, init_func=init, blit=True, interval=interval, repeat=False
    )
    # Optionally save
    if save_name:
        try:
            writer = animation.FFMpegWriter(fps=30)
            ani.save(save_name, writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
            print(f"Saved animation to {save_name}")
        except Exception as e:
            print('Failed to save MP4 (ffmpeg missing?). Error:', e)
    plt.show()

if __name__ == '__main__':
    # Run the advection animation for two polynomial orders and show progress
    N_vec = [5, 25]
    tf = 2.0
    c = 1.0  # advection speed in exact solution
    cfl_target = 0.1  # user-facing CFL number (tuneable)

    # Use fixed timestep as requested
    dt = 0.0002
    print(f"Using fixed dt = {dt:.6e}")

    # Square wave ICs
    u0 = lambda x: np.where(abs(x) < 0.5, 1.0, 0.0)
    f = lambda x, t: 0

    # Exact solution for comparison (periodic, shift by ct)
    exact_sol = lambda x, t: u0(np.mod(x - c * t + 1, 2) - 1)

    # Compute solutions for each N with progress
    solutions = []
    times = None
    for N in tqdm(N_vec, desc='Computing solutions for N values'):
        (x_pts, _) = gll_pts_wts(N)
        u0_vec_fn = lambda x_val: u0(x_val)
        print(f"Solving for N = {N} (showing progress)")
        U = solve_1d_advection(f, N, u0_vec_fn, dt, tf, show_progress=True, c=c)
        solutions.append({'N': N, 'x_pts': x_pts, 'U': U})
        if times is None:
            times = np.arange(dt, tf + dt, dt)

    # Prepare animation grid
    x_vals = np.linspace(-1, 1, 400)

    # Determine y-limits from analytic square wave (1.5x amplitude)
    exact0 = exact_sol(x_vals, 0)
    mean_val = 0.5 * (np.max(exact0) + np.min(exact0))
    amp = max(abs(np.max(exact0) - mean_val), abs(np.min(exact0) - mean_val))
    y_lower = mean_val - 1.5 * amp
    y_upper = mean_val + 1.5 * amp

    # Build figure and lines
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = []
    line_exact, = ax.plot([], [], label='Exact', color='black', linewidth=2.5)
    lines.append(line_exact)
    all_linestyles = ['--', '-.', (0, (5, 5)), (0, (3, 5, 1, 5))]
    for i, sol in enumerate(solutions):
        ln, = ax.plot([], [], label=f"N = {sol['N']}", linestyle=all_linestyles[i % len(all_linestyles)])
        lines.append(ln)

    ax.set_xlim(-1, 1)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x,t)$')
    ax.grid(True)
    fig.subplots_adjust(bottom=0.20)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(solutions) + 1)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def animate(i):
        t_val = times[i]
        x_fine = x_vals
        exact_vals = exact_sol(x_fine, t_val)
        lines[0].set_data(x_fine, exact_vals)
        for k, sol in enumerate(solutions, start=1):
            coeffs = sol['U'][i]
            # solutions store reduced vectors (last periodic node removed), so use x_pts without the final node
            approx_fn = construct_solution(coeffs, sol['x_pts'][:-1])
            approx_vals = [approx_fn(xi) for xi in x_fine]
            lines[k].set_data(x_fine, approx_vals)
        ax.set_title(f"$\\partial_t u + \\partial_x u = 0 \\quad t = {t_val:.3f}$")
        return lines

    total_frames = len(times)
    interval_ms = int(10000 / total_frames) if total_frames > 0 else 50
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init, blit=False, interval=interval_ms, repeat=False)

    # Save to MP4
    try:
        writer = animation.FFMpegWriter(fps=30)
        ani.save('advection_multiN.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
        print('Saved advection animation to advection_multiN.mp4')
    except Exception as e:
        print('Failed to save MP4 (ffmpeg missing?). Error:', e)

    plt.show()
