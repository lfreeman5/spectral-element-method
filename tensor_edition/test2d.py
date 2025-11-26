from tensor_sem_utils import create_mass_stiffness_2d, map_1d_to_2d, map_2d_to_1d, modify_lhs_rhs_dirichlet
from gll_utils import gll_pts_wts
import numpy as np
from plotting_utils import plot_solution_2d

if __name__ == "__main__":
    # Define N
    # Create mass, stiffness matrices, which will be N2 x N2
    # Create 2D u0 
    alpha = 1
    N = 10
    pts, wts = gll_pts_wts(N)
    M,A = create_mass_stiffness_2d(N)
    A*=-alpha # Add diffusion effect. What is correct sign?

    Nt = 100
    dt = 0.01
    U = np.zeros((Nt,N+1,N+1))
    U[0,:,:] = 1. # Uniform initial starting condition

    for n in range(Nt-1):
        print(f'Calculating iter {n+1}, previous max: {np.max(U[n,:,:])}')
        LHS = M/dt-A
        RHS = (1/dt*M)@(map_2d_to_1d(U[n,:,:],N))
        lmod, rmod = modify_lhs_rhs_dirichlet(LHS, RHS, N, 0.0)
        U[n+1,:,:] = map_1d_to_2d(np.linalg.solve(lmod,rmod),N)
        if(n%10==0):
            # Plot the current state every 10 iterations
            plot_solution_2d(U, np.arange(Nt)*dt, pts, (n+1)*dt)

