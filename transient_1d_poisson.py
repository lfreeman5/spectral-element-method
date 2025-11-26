import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from sem_utils import modify_A_b_dirichlet, construct_a_matrix, construct_b_vector, construct_m_matrix
import sympy as sp
import time
from tqdm import tqdm

'''
    Uses the Spectral element method to solve:
        d²u/dx² = f(x),   x ∈ [−1, 1]

    Nomenclature:
    N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials        
'''
pi = np.pi

# Problem Constants: 
x_a, x_b = -1, 1 # x_a<=x<=x_b. Assumed to be constant and x ∈ [−1, 1]. This requirement will be removed later.

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


def compare_exact_approx_animation(exact, U_solution, N, dt, t_f):
    import matplotlib.animation as animation

    (gll_pts, _) = gll_pts_wts(N)
    times = np.arange(dt, t_f+dt, dt)
    x_vals = np.linspace(x_a, x_b, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    line_exact, = ax.plot([], [], label='Exact', color='blue')
    line_approx, = ax.plot([], [], label='Approx', linestyle='--', color='red')
    ax.set_xlim(x_a, x_b)
    # Fixed y-limits requested: always show [-1.0, 1.5]
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')
    ax.grid(True)
    ax.legend(loc='best')

    def init():
        line_exact.set_data([], [])
        line_approx.set_data([], [])
        return line_exact, line_approx

    def animate(i):
        t_val = times[i]
        coeffs = U_solution[i]
        approx = construct_solution(coeffs, gll_pts)
        exact_vals = [exact(x, t_val) for x in x_vals]
        approx_vals = [approx(x) for x in x_vals]
        line_exact.set_data(x_vals, exact_vals)
        line_approx.set_data(x_vals, approx_vals)
        ax.set_title(f"t = {t_val:.2f}")
        return line_exact, line_approx

    # Calculate interval so animation lasts 5 seconds (5000 ms) -- half the previous duration
    total_frames = len(times)
    interval_ms = int(5000 / total_frames) if total_frames > 0 else 25

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()

if __name__ == '__main__':
    # Time and problem setup
    tf = 20.0
    dt = 0.02

    x, t = sp.Symbol('x'), sp.Symbol('t')
    u_exact = sp.sin(t / (x + 2))  # Exact solution
    f = sp.diff(u_exact, t, 1) - sp.diff(u_exact, x, 2)  # Compute corresponding forcing function

    # Print the problem being solved
    print("Solving the transient 1D Poisson equation (diffusion equation):")
    print("    du/dt - d²u/dx² = f(x, t)")
    print(f"with exact solution: u(x, t) = {u_exact}")

    # Prepare printable forcing forms (LaTeX + python); round numeric literals for readability
    try:
        latex_f = sp.latex(f)
    except Exception:
        latex_f = str(f)
    import re
    def _round_number_match(m):
        s = m.group(0)
        try:
            val = float(s)
            return f"{val:.2f}"
        except Exception:
            return s
    latex_f_rounded = re.sub(r"-?\d+\.?\d*", _round_number_match, latex_f)
    print("Forcing function f (LaTeX, rounded):", latex_f_rounded)

    exact = sp.lambdify((x, t), u_exact, 'numpy')
    forcing_func = sp.lambdify((x, t), f, 'numpy')

    # Choose a small set of polynomial orders to animate together
    # N_vec = [5, 10, 15]
    N_vec = [5, 15]

    # Compute transient solutions for each N and store data for animation
    solutions = []  # list of dicts: {'N', 'gll_pts', 'U'}
    times = None
    for N in tqdm(N_vec, desc="Computing transient solutions"):
        (gll_pts, gll_wts) = gll_pts_wts(N)
        u_0 = lambda x_val: exact(x_val, 0)
        u_L = lambda t_val: exact(x_a, t_val)
        u_R = lambda t_val: exact(x_b, t_val)
        print(f"Computing transient solution for N={N} ...")
        U_sol = transient_solution(forcing_func, N, u_L, u_R, u_0, dt, tf)
        solutions.append({'N': N, 'gll_pts': gll_pts, 'U': U_sol})
        if times is None:
            times = np.arange(dt, tf + dt, dt)

    # Prepare animation grid
    x_vals = np.linspace(x_a, x_b, 200)
    total_frames = len(times)

    # Plot setup: one line for exact, one for each N in N_vec
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = []
    # exact line
    line_exact, = ax.plot([], [], label='Exact', color='black', linewidth=2.5)
    lines.append(line_exact)
    # styles for approximations (wrap if needed)
    all_linestyles = ["solid", (0, (5, 5)), "dashdot", (0, (3, 5, 1, 5)), (0, (1, 5)), (0, (1, 3)), (0, (1, 1))]
    for i, sol in enumerate(solutions):
        style = all_linestyles[i % len(all_linestyles)]
        ln, = ax.plot([], [], label=f"N = {sol['N']}", linestyle=style)
        lines.append(ln)

    ax.set_xlim(x_a, x_b)
    # Fixed y-limits requested: always show [-1.0, 1.5]
    vals = []
    for i, sol in enumerate(solutions):
        # evaluate first and last frames to get range approximation
        U0 = sol['U'][0]
        U1 = sol['U'][-1]
        vals.append(np.min(U0))
        vals.append(np.max(U0))
        vals.append(np.min(U1))
        vals.append(np.max(U1))
    # include exact at t=0 and t=tf
    exact0 = [exact(xi, 0) for xi in x_vals]
    exactf = [exact(xi, tf) for xi in x_vals]
    vals.append(min(exact0))
    vals.append(max(exact0))
    vals.append(min(exactf))
    vals.append(max(exactf))
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x,t)$')
    ax.grid(True)
    # Make room at the bottom for a legend and create a figure-level legend below the axes
    fig.subplots_adjust(bottom=0.20)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(solutions)+1)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def animate(i):
        t_val = times[i]
        # exact
        exact_vals = [exact(xi, t_val) for xi in x_vals]
        lines[0].set_data(x_vals, exact_vals)
        # approximations
        for k, sol in enumerate(solutions, start=1):
            coeffs = sol['U'][i]
            approx_fun = construct_solution(coeffs, sol['gll_pts'])
            approx_vals = [approx_fun(xi) for xi in x_vals]
            lines[k].set_data(x_vals, approx_vals)
        # Put the whole title inside a math environment so \quad is interpreted as LaTeX spacing
        ax.set_title("$\\dfrac{d^2 u}{dx^2} = " + latex_f_rounded + f" \\quad t = {t_val:.2f}$")
        return lines

    # Calculate interval so animation lasts 5 seconds (5000 ms) -- half the previous duration
    interval_ms = int(5000 / total_frames) if total_frames > 0 else 25
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init, blit=False, interval=interval_ms, repeat=False)

    # Save to MP4 (requires ffmpeg)
    try:
        writer = animation.FFMpegWriter(fps=15)
        ani.save('transient_multiN.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
        print('Saved transient animation to transient_multiN.mp4')
    except Exception as e:
        print('Failed to save MP4 (ffmpeg missing?). Error:', e)

    plt.show()
    # keep reference
    _ani_ref = ani


