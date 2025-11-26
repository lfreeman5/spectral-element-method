from tensor_sem_utils import create_mass_stiffness_2d, map_1d_to_2d, map_2d_to_1d, modify_lhs_rhs_dirichlet, create_C
from gll_utils import gll_pts_wts
import numpy as np
from plotting_utils import plot_solution_2d

if __name__ == "__main__":
    # Define N
    # Create mass, stiffness matrices, which will be N2 x N2
    # Create 2D u0 
    alpha = .001
    N = 10
    pts, wts = gll_pts_wts(N)
    M,A = create_mass_stiffness_2d(N)
    A*=-alpha # Add diffusion effect. What is correct sign?

    Nt = 100
    dt = 0.01
    U = np.zeros((Nt,N+1,N+1))
    # Gaussian bump initial condition, not centered at origin
    x0, y0 = 0.5, 0.0
    sigma = 0.1
    X, Y = np.meshgrid(pts, pts, indexing='ij')
    U[0,:,:] = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    # Deal with advection field
    M_overintegrated = 15
    cx = lambda x,y: -y
    cy = lambda x,y: x
    mpts, _ = gll_pts_wts(M_overintegrated)
    CxMM, CyMM = np.zeros((M_overintegrated+1,M_overintegrated+1)), np.zeros((M_overintegrated+1,M_overintegrated+1))
    for i in range(M_overintegrated+1):
        for j in range(M_overintegrated+1):
            CxMM[i,j] = cx(mpts[i],mpts[j])
            CyMM[i,j] = cy(mpts[i],mpts[j])
    C = create_C(N,M_overintegrated,CxMM,CyMM)

    for n in range(Nt-1):
        print(f'Calculating iter {n+1}, previous max: {np.max(U[n,:,:])}')
        LHS = M/dt-A-C
        RHS = (1/dt*M)@(map_2d_to_1d(U[n,:,:],N))
        lmod, rmod = modify_lhs_rhs_dirichlet(LHS, RHS, N, 0.0)
        U[n+1,:,:] = map_1d_to_2d(np.linalg.solve(lmod,rmod),N)
        if(n%10==0):
            # Plot the current state every 10 iterations
            plot_solution_2d(U, np.arange(Nt)*dt, pts, (n+1)*dt)

