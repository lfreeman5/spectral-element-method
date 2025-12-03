import numpy as np
from gll_utils import gll_pts_wts
from plotting_utils import plot_ns_solution_2d
from tensor_sem_utils import create_mass_stiffness_2d, map_1d_to_2d, map_2d_to_1d, modify_lhs_rhs_dirichlet, create_Jhat, create_Dtilde, create_Bhat_Dhat
from ns_utils import calc_v_hat, pressure_solve, correct_vhat_with_pressure, helmholtz_update, curlcurl




if __name__ == "__main__":

    # Define N
    # Create mass, stiffness matrices, which will be N2 x N2
    # Create 2D u0 
    alpha = .001
    N = 10
    pts, wts = gll_pts_wts(N)
    M,A = create_mass_stiffness_2d(N)
    
    k = 3 # time stepping order for BDFk/ABk 
    
    Nt = 100
    dt = 0.01


    u = np.zeros((Nt,(N+1)**2))
    v = np.zeros((Nt,(N+1)**2))

    # set initial velocity conditions (set first few time steps as the same)
    u[0:k,:] = ?
    v[0:k,:] = ?


    # Deal with advection field
    M_over = 15 # Overintegrated velocity field
    mpts, _ = gll_pts_wts(M_over)

    # calc matrices that are constant
    J_hat = create_Jhat(N,M_over) 
    B_hat_M, D_hat = create_Bhat_Dhat(M_over)
    D_tilde = create_Dtilde(N,M_over)
    B_M = np.kron(B_hat_M, B_hat_M)

    for n in range(k-1, Nt-1):
        print(f'Calculating iter {n+1}, previous max u: {np.max(u[n,:])}, previous max v: {np.max(v[n,:])}')

        # update v_hat using saved u and v data,   if k=3, (n-k+1):(n+1) should give n-2, n-1, n
        v_hat = calc_v_hat(k, dt, u[(n-k+1):(n+1), ::-1], v[(n-k+1):(n+1), ::-1])

        # do pressure solve
        p = pressure_solve(N, k, dt, vel, v_hat, A, M, Dx, Dy, vel_boundary)

        # do helmholtz solves
        print("Helmholtz solve u")
        u[n+1,:] = 
        print("Helmholtz solve v")
        v[n+1,:] = 

    
        if(n%10==0):
            # Plot the current state every 10 iterations
            plot_ns_solution_2d(map_1d_to_2d(u[n+1,:],N), map_1d_to_2d(v[n+1,:],N), np.arange(Nt)*dt, pts, (n+1)*dt)
