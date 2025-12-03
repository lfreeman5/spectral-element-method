from tensor_sem_utils import create_mass_stiffness_2d, map_1d_to_2d, map_2d_to_1d, modify_lhs_rhs_dirichlet, create_C, construct_neumann_rhs_2d, modify_lhs_rhs_mixed, construct_advection_flux_rhs_2d
from gll_utils import gll_pts_wts
import numpy as np
from plotting_utils import plot_solution_2d

if __name__ == "__main__":
    alpha = 0.001
    N = 10
    pts, wts = gll_pts_wts(N)
    M, A = create_mass_stiffness_2d(N)
    A *= -alpha

    Nt = 1000
    dt = 0.01
    U = np.zeros((Nt, N+1, N+1))

    # Initial Gaussian
    x0, y0 = -0.5, 0.0
    sigma = 0.1
    X, Y = np.meshgrid(pts, pts, indexing='ij')
    U[0,:,:] = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    M_over = 15
    # Advection velocity (rotational)
    # cx = lambda x,y: -y
    # cy = lambda x,y: x

    # Advection velocity (translational)
    cx = lambda x,y: -1
    cy = lambda x,y: 0

    # Volume advection operator
    mpts, _ = gll_pts_wts(M_over)
    CxMM = np.zeros((M_over+1, M_over+1))
    CyMM = np.zeros((M_over+1, M_over+1))
    for i in range(M_over+1):
        for j in range(M_over+1):
            CxMM[i,j] = cx(mpts[i], mpts[j])
            CyMM[i,j] = cy(mpts[i], mpts[j])
    C = create_C(N, M_over, map_2d_to_1d(CxMM, M_over), map_2d_to_1d(CyMM, M_over))

    # All-Dirichlet boundaries
    g_bc = lambda x,y: -1.0  # advecting inflow condition
    zero_flux = lambda x,y: 0.0  # homogeneous Dirichlet
    BCs = {
        'left':   ('dirichlet', zero_flux),
        'right':  ('dirichlet', zero_flux),
        'bottom': ('dirichlet', zero_flux),
        'top':    ('dirichlet', zero_flux),
    }
    # Time stepping
    for n in range(Nt-1):
        u_current = U[n,:,:]
        print(f'Calculating iter {n+1}, previous max: {np.max(abs(U[n,:,:]))}')
        # EXPLICIT advection:
        LHS = M/dt - A - C 
        # LHS = M/dt - A 
        u_vec = map_2d_to_1d(u_current, N)
        RHS = (1/dt*M) @ (u_vec)
        # Mixed BCs (here all Dirichlet)
        LHS_mod, RHS_mod = modify_lhs_rhs_mixed(LHS, RHS, N, BCs, alpha, pts, wts,u_field=u_current,c=(cx, cy))
        # Solve and store
        u_next_vec = np.linalg.solve(LHS_mod, RHS_mod)
        U[n+1,:,:] = map_1d_to_2d(u_next_vec, N)

        if n % 10 == 0:
            plot_solution_2d(U, np.arange(Nt)*dt, pts, (n+1)*dt)