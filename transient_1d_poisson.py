import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
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

def construct_a_matrix(N): # Only assembled once, so cost is amortized for transient problem
    A = np.zeros((N+1, N+1))
    (gll_pts, _) = gll_pts_wts(N)
    start_time = time.time()
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
            A[i, j] = integrate_gll(x_a, x_b, Lij_prime, N) - L_j_prime(1) * L_i(1) + L_j_prime(-1) * L_i(-1)
        # print(f"Time to compute row {i} of A: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
    return A

def construct_m_matrix(N):
    (_, gll_wts) = gll_pts_wts(N)
    M = np.diag(gll_wts)
    return M

def construct_b_vector(N,f,t_eval):
    '''
    Constructs the load vector b for the 1D Poisson equation.
    Arguments:
        N - polynomial order (number of intervals is N, number of points is N+1)
        f - forcing function, callable, f=f(x,t)
    Returns:
        b - load vector of length N+1
    '''
    b = np.zeros(N+1)
    (gll_pts,_) = gll_pts_wts(N)
    for i in range(N+1):
        L_i = create_lagrange_poly(i,gll_pts)
        f_L_i = lambda x: L_i(x)*f(x,t_eval)
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
        f_vec = construct_b_vector(N,f,t) # evaluated at next timestep which is t (specific to backward euler)
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
    ax.set_ylim(np.min(U_solution), np.max(U_solution))
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

    # Calculate interval so animation lasts 10 seconds (10000 ms)
    total_frames = len(times)
    interval_ms = int(10000 / total_frames) if total_frames > 0 else 50

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()

if __name__ == '__main__':
    N=15
    (gll_pts, gll_wts) = gll_pts_wts(N)
    tf = 30
    dt = 0.005

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
    u_L = lambda t_val: exact(x_a, t_val)
    u_R = lambda t_val: exact(x_b, t_val)

    U_solution = transient_solution(forcing_func, N, u_L, u_R, u_0, dt, tf)

    comparison_t_idx = int((tf-10*dt) // dt)
    compare_exact_approx(lambda x_val: exact(x_val, dt*comparison_t_idx), 
                         construct_solution(U_solution[comparison_t_idx,:],gll_pts))
    compare_exact_approx_animation(exact, U_solution, N, dt, tf)


