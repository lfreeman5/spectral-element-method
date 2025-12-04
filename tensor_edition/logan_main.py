import numpy as np
from gll_utils import gll_pts_wts
from plotting_utils import plot_ns_solution_2d_vector, plot_pressure_2d
from tensor_sem_utils import create_mass_stiffness_2d, map_1d_to_2d, map_2d_to_1d, modify_lhs_rhs_dirichlet, create_Jhat, create_Dtilde, create_Bhat_Dhat
from ns_utils import calc_v_hat, pressure_solve, correct_vhat_with_pressure, helmholtz_update, curlcurl, evaluate_cfl




if __name__ == "__main__":

    # Define N
    # Create mass, stiffness matrices, which will be N2 x N2
    # Create 2D u0 
    alpha = .00001
    N = 9
    pts, wts = gll_pts_wts(N)
    M,A = create_mass_stiffness_2d(N)
    
    k = 3 # time stepping order for BDFk/ABk 
    
    Nt = 50000
    dt = 0.01

    lid_velocity = 100.0


    vel = np.zeros((Nt,2,(N+1)**2))

    # set initial velocity conditions (set first few time steps as the same)
    u_init = np.zeros((N+1,N+1))
    v_init = np.zeros((N+1,N+1))
    # u_init[4:6, 4:6] = 0.1
    # v_init[1:N, 1:N] = 0.1

    vel[0:k,0,:] = map_2d_to_1d(u_init, N)
    vel[0:k,1,:] = map_2d_to_1d(v_init, N)

    vel_boundary = np.zeros((4,2))
    vel_boundary[1,0] = lid_velocity


    # Deal with advection field
    M_over = 15 # Overintegrated velocity field
    mpts, _ = gll_pts_wts(M_over)

    # calc matrices that are constant
    eye = np.eye(N+1)
    J_hat = create_Jhat(N,M_over) 
    B_hat_M, __ = create_Bhat_Dhat(M_over)
    __, D_hat_N = create_Bhat_Dhat(N)
    D_tilde = create_Dtilde(N,M_over)
    B_M = np.kron(B_hat_M, B_hat_M)
    Dx = np.kron(eye,D_hat_N) 
    Dy = np.kron(D_hat_N,eye) 

    for n in range(k-1, Nt-1):
        if(n%100==0):
            print(f'Calculating iter {n+1}, previous max u: {np.max(vel[n,0,:])}, previous max v: {np.max(vel[n,1,:])}')
            cfl = evaluate_cfl(N, dt, vel[n,:,:])
            print("cfl = ", cfl )
        # update v_hat using saved u and v data,   if k=3, (n-k+1):(n+1) should give n-2, n-1, n
        v_hat = calc_v_hat(k, dt, vel[(n-k+1):(n+1)][::-1], M, J_hat, B_M, D_tilde)
        # print("n, n-1, n-2", vel[(n-k+1):(n+1)][::-1])
        # print("n-2, n-1, n", vel[(n-k+1):(n+1),:,:])

        # do pressure solve
        # print("Pressure solve")
        p = pressure_solve(N, k, dt, vel[(n-k+1):(n+1)][::-1], v_hat, A, M, Dx, Dy, vel_boundary)
        v_hathat = correct_vhat_with_pressure(N,dt,v_hat, p, Dx, Dy)

        # do helmholtz solves
        # print("Helmholtz solves")
        vel[n+1,:,:] = helmholtz_update(N,dt, k, alpha, A, M, v_hathat, vel_boundary) 



        if(n>8000 and n%1000==0):
            # Plot the current state every 10 iterations
            u2d = map_1d_to_2d(vel[n+1,0,:], N)
            v2d = map_1d_to_2d(vel[n+1,1,:], N)
            plot_ns_solution_2d_vector(u2d, v2d, np.arange(Nt)*dt, pts, (n+1)*dt)

            plot_pressure_2d(map_1d_to_2d(p, N), np.arange(Nt)*dt, pts, (n+1)*dt)
