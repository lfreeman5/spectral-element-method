# import numpy as np
# import matplotlib.pyplot as plt
# from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
# from gll_utils import gll_pts_wts, integrate_gll, print_matrix
# from sem_utils import construct_a_matrix, construct_b_vector, modify_A_b_dirichlet
# import sympy as sp
# import time
# from tqdm import tqdm
# '''
#     Uses the Spectral element method to solve:
#         d²u/dx² = f(x),   x ∈ [−1, 1]

#     Nomenclature:
#     N refers to the polynomial order. The Lagrange polynomials are L0...LN ie N+1 Lagrange polynomials        
# '''
# pi = np.pi

# # Problem Constants: 
# x_a, x_b = -1, 1 # x_a<=x<=x_b. Assumed to be constant and x ∈ [−1, 1]. This requirement will be removed later.

# def solve_1d_poisson(f,N,u_L,u_R):
#     '''
#         Solves the 1D poisson equation:
#             d²u/dx² = f(x),   x ∈ [−1, 1], u(-1)=u_L, u(1)=u_R
#         using order-N Lagrange polynomials on GLL points
#         Arguments:
#             f - the forcing function. Callable.
#             N - polynomial order
#             u_L - LHS dirichlet BC
#             u_R - RHS dirichlet BC
#         Returns:
#         solution - callable function of x
#     '''
#     A = construct_a_matrix(N)
#     b = construct_b_vector(N, f)
#     A, b = modify_A_b_dirichlet(A, b, u_L, u_R)
#     u = np.linalg.solve(A,b)
#     solution = construct_solution(u, gll_pts_wts(N)[0])
#     return solution

# def compare_exact_approx(exact, approx, N, iteration):
#     x_vals = np.linspace(x_a, x_b, 200)
#     exact_vals = [exact(x) for x in x_vals]
#     approx_vals = [approx(x) for x in x_vals]
#     all_linestyles = [
#         "solid",            # solid
#         (0, (5, 5)),        # dashed
#         "dashdot",          # dash-dot (standard Matplotlib style)
#         (0, (3, 5, 1, 5)),  # dash-dot with spacing
#         (0, (1, 5)),        # dotted (sparse)
#         (0, (1, 3)),        # dotted (denser)
#         (0, (1, 1))         # very fine dotted
#     ]
#     if iteration == 0:
#         plt.plot(x_vals, exact_vals, label='Exact Solution', color='red')
#     # choose style by iteration (wrap if more curves than styles)
#     style = all_linestyles[iteration % len(all_linestyles)]
#     plt.plot(x_vals, approx_vals, label='N = ' + str(N), linestyle=style, color='black')
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$u(x)$')
#     plt.grid(True)
#     # plt.legend(loc='best')
#     # move legend below the plot
#     plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=3)
#     # plt.show()

# if __name__ == '__main__':
#     plt.rcParams["font.family"] = "Serif"
#     plt.rcParams["font.size"] = 17.0
#     plt.rcParams["axes.labelsize"] = 17.0
#     plt.rcParams['lines.linewidth'] = 1.0 # 1.0
#     plt.rcParams["xtick.minor.visible"] = True 
#     plt.rcParams["ytick.minor.visible"] = True
#     plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
#     plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"]= True 
#     plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] =True
#     plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
#     plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
#     plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
#     plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
#     plt.rcParams["mathtext.fontset"] = "dejavuserif"
#     plt.rcParams['figure.dpi'] = 300.0
#     ## change legend parameters
#     plt.rcParams["legend.fontsize"] = 17.0
#     plt.rcParams["legend.frameon"] = True
#     subdict = {"figsize" : (3.25,3.5),"constrained_layout" : True,"sharex" : True}
#     N=5

#     x = sp.Symbol('x')
#     u_exact = sp.sin(1/(x+1.1)) # Exact solution
#     f = -sp.diff(u_exact,x,2) # Compute corresponding forcing function # the 2 is for second derivative. the negative is because we are moving it to RHS. 
#     exact = sp.lambdify(x,u_exact,'numpy')
#     forcing_func = sp.lambdify(x,f,'numpy')
#     u_L, u_R = exact(x_a), exact(x_b)
#     N_vec = [5, 10, 15, 20, 25, 30, 35]
#     # N_vec = [5, 6,7,8,9,10,11]
#     iteration = 0
#     plt.figure(figsize=(8, 5))
#     for N in tqdm(N_vec, desc="Solving 1D Poisson"):
#         u_approx = solve_1d_poisson(forcing_func, N, u_L, u_R)
#         compare_exact_approx(exact, u_approx, N, iteration)
#         iteration += 1
#     # save the plot as an svg with 3 columns legend
#     # Make the forcing function available in several forms and put it in the title.
#     try:
#         from sympy.printing import pycode
#         latex_f = sp.latex(f)
#         py_f = pycode(f)
#     except Exception:
#         # Fallback to string forms if printing helpers are not available
#         latex_f = str(f)
#         py_f = str(f)

#     # Truncate numeric literals in the LaTeX string to 2 decimal places for readability
#     import re
#     def _round_number_match(m):
#         s = m.group(0)
#         try:
#             val = float(s)
#             return f"{val:.2f}"
#         except Exception:
#             return s

#     # replace occurrences of integers or decimals (simple approach)
#     latex_f_rounded = re.sub(r"-?\d+\.?\d*", _round_number_match, latex_f)

#     # Print out the forcing function forms so the user can copy/paste the python/numpy form
#     # print("Forcing function f (sympy):", f)
#     # print("Forcing function f (LaTeX, rounded):", latex_f_rounded)
#     # print("Forcing function f (python/numpy):", py_f)

#     # Set plot title showing the differential equation in LaTeX (single math environment)
#     plt.title(f"$\\dfrac{{d^2 u}}{{dx^2}} = $f(x)$ = {latex_f_rounded}$")
#     plt.savefig('steady_1d_poisson_solution.svg', bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
from lagrange_utils import create_lagrange_derivative, create_lagrange_poly, construct_solution, create_lagrange_derivative_gll_points
from gll_utils import gll_pts_wts, integrate_gll, print_matrix
from sem_utils import construct_a_matrix, construct_a_matrix_Neumann, construct_b_vector, modify_A_b_dirichlet, modify_A_b_neumann
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

def solve_1d_poisson_neumann(f,N,deriv_L,deriv_R):
    '''
        Solves the 1D poisson equation:
            d²u/dx² = f(x),   x ∈ [−1, 1], u(-1)=u_L, u(1)=u_R
        using order-N Lagrange polynomials on GLL points
        Arguments:
            f - the forcing function. Callable.
            N - polynomial order
            deriv_L - LHS Neumann BC (u'(-1))
            deriv_R - RHS Neumann BC (u'(1))
        Returns:
        solution - callable function of x
    '''
    A = construct_a_matrix_Neumann(N)
    b = construct_b_vector(N, f)
    A, b = modify_A_b_neumann(A, b, deriv_L, deriv_R, enforce_mean=True)
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
    N=15
    # x = sp.Symbol('x')
    # u_exact = sp.sin(1/(x+1.1)) # Exact solution
    # f = -sp.diff(u_exact,x,2) # Compute corresponding forcing function
    # exact = sp.lambdify(x,u_exact,'numpy')
    # forcing_func = sp.lambdify(x,f,'numpy')
    # u_L, u_R = exact(x_a), exact(x_b)
    # deriv_L, deriv_R = sp.diff(u_exact,x).subs(x,x_a), sp.diff(u_exact,x).subs(x,x_b) # sp.diff(u_exact,x) is the derivative of u_exact with respect to x. The .subs means x is replaced with x_a or x_b respectively.
    # u_approx = solve_1d_poisson(forcing_func, N, u_L, u_R)
    # u_approx_neumann = solve_1d_poisson_neumann(forcing_func, N, deriv_L, deriv_R)
    # compare_exact_approx(exact, u_approx)
    # compare_exact_approx(exact, u_approx_neumann)
    
    x = sp.Symbol('x')

    # Try this smooth, non-trivial Neumann test:
    u_exact = sp.cos(sp.pi * x)
    f = -sp.diff(u_exact, x, 2)

    exact = sp.lambdify(x, u_exact, 'numpy')
    forcing_func = sp.lambdify(x, f, 'numpy')
    u_L, u_R = exact(x_a), exact(x_b)
    deriv_L, deriv_R = sp.diff(u_exact, x).subs(x, x_a), sp.diff(u_exact, x).subs(x, x_b)  # sp.diff(u_exact,x) is the derivative of u_exact with respect to x. The .subs means x is replaced with x_a or x_b respectively.
    u_approx = solve_1d_poisson(forcing_func, N, u_L, u_R)
    u_approx_neumann = solve_1d_poisson_neumann(forcing_func, N, deriv_L, deriv_R)
    compare_exact_approx(exact, u_approx)
    compare_exact_approx(exact, u_approx_neumann)
