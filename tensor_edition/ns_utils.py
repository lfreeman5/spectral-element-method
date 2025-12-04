import numpy as np
from tensor_sem_utils import *
from fractions import Fraction


def calc_v_hat(k, dt, saved_vel_coefs, M, J_hat, B_M, D_tilde):
    '''
    calculates v_hat used in pressure poisson solve, based on eq. 6.5.8 in the text
    k: order of BDF (implicit) terms and Adams Bashforth (explicit) terms
    saved_u_coefs: (k,2,(N+1)^2) array with k sets of previous [u,v] data timesteps [n, n-1, n-2, ...]
    J_hat: (M+1,N+1) interpolation matrix
    B_M: ((M+1)^2, (M+1)^2) array, mass matrix for overintegration
    D_tilde: (M+1,N+1) array, derivative matrix for overintegration
    '''
    # get integration coefficients
    bj = AB_coefs(k)
    __, beta_k_minus_j = BDFk_coefs(k)

    v_hat_u = np.zeros(len(saved_vel_coefs[0,0]))
    v_hat_v = np.zeros(len(saved_vel_coefs[0,0])) 
    for j in range(k):
    #     print("      bj = ", Fraction(bj[j]).limit_denominator())
    #     print("beta k-j = ", Fraction(beta_k_minus_j[j]).limit_denominator())


        Cu, Cv = nonlinear_advection_at_previous_time(saved_vel_coefs[j,0,:], saved_vel_coefs[j,1,:], J_hat, B_M, D_tilde) # j = 0 is
        v_hat_u += -beta_k_minus_j[j]*saved_vel_coefs[j,0,:] + dt*bj[j]*Cu
        v_hat_v += -beta_k_minus_j[j]*saved_vel_coefs[j,1,:] + dt*bj[j]*Cv

    return np.array([v_hat_u, v_hat_v]) # Change shape to be 2 x (N+1)^2 

def pressure_solve(N,k,dt,vel,vhat,A,M,Dx,Dy,vel_boundary):
    '''
    Solves the pressure Poisson equation for incompressible flow.
    N: int, polynomial order
    k: int, time integration order
    dt: float, timestep
    vel: (k, 2, (N+1)^2) array, velocity coefficients [u, v]
    vhat: (2, (N+1)^2) array, v_hat coefficients [v_hat_x, v_hat_y]
    A: ((N+1)^2, (N+1)^2) array, stiffness matrix
    M: ((N+1)^2, (N+1)^2) array, mass matrix
    Dx, Dy: ((N+1)^2, (N+1)^2) arrays, differentiation tensors
    vel_boundary: (4, 2) array, velocity boundary conditions for each side of the domain.
        The order is [bottom, top, left, right], with each entry being a 2D velocity vector.
    Returns p: ((N+1)^2,) array, pressure coefficients
    '''

    # Set up boundary term 
    B = np.zeros(((N+1),(N+1)))

    # GLL pts, wts
    _,wts = gll_pts_wts(N)

    # Time integration coeffs
    expl_coeffs = AB_coefs(k)
    impl_coeffs = BDFk_coefs(k)

    # Compute div(vhat)/dt
    div_vhat = Dx@vhat[0]+Dy@vhat[1]

    # Compute curl(curl(v^{n+1-j})) for non-boundary-dependent part
    inner_boundary_term = np.zeros((2,(N+1)**2))
    for j in range(k):
        inner_boundary_term += expl_coeffs[j]*curlcurl(vel[j],Dx,Dy) #
    inner_boundary_term -= vhat/dt
    inner_boundary_u = map_1d_to_2d(inner_boundary_term[0],N)
    inner_boundary_v = map_1d_to_2d(inner_boundary_term[1],N)
    inner_boundary_term = np.array([inner_boundary_u, inner_boundary_v])
    # inner_boundary_term = map_1d_to_2d(inner_boundary_term, N)
    # At this point need to move to side-by-side based on velocity in
    # Bottom: 
    # Everything is (2x(N+1)**2)
    bottom_normal = np.array([0,-1])
    bottom_term = inner_boundary_term[:,:,0] + np.tile(vel_boundary[0], (N+1,1)).T/dt # shape (2, N+1)
    bottom_dotted = -np.dot(bottom_normal, bottom_term) # shape (N+1,)
    B[:,0] += bottom_dotted*wts

    # Right: 
    right_normal = np.array([1,0])
    right_term = inner_boundary_term[:,-1,:] + np.tile(vel_boundary[3], (N+1,1)).T/dt # shape (2, N+1)
    right_dotted = -np.dot(right_normal, right_term) # shape (N+1,)
    B[-1,:] += right_dotted*wts

    # Top: 
    top_normal = np.array([0,1])
    top_term = inner_boundary_term[:,:, -1] + np.tile(vel_boundary[1], (N+1,1)).T/dt # shape (2, N+1)
    top_dotted = -np.dot(top_normal, top_term) # shape (N+1,)
    B[:,-1] += -top_dotted*wts

    # Left: 
    left_normal = np.array([-1,0])
    left_term = inner_boundary_term[:,0,:] + np.tile(vel_boundary[2], (N+1,1)).T/dt # shape (2, N+1)
    left_dotted = -np.dot(left_normal, left_term) # shape (N+1,)
    B[0,:] += -left_dotted*wts
    # Then assemble the boundary condition with constant g. Then map full thing 1d-to-2d and extract the relevant side
    # ^^^ That is horribly inefficient

    # Then we move to the pressure solve
    # Solve (A,b) where A is stiffness, b=f_ij+boundary
    # assemble f_ij based on div vhat, I think M kron div vhat? 
    F = M@div_vhat/dt
    LHS = A # Check sign on this - should incorporate the negative in constructing it
    B*=0 # Inviscid pressure condition
    RHS = F + map_2d_to_1d(B,N)
    p = np.linalg.solve(LHS, RHS)

    pmean = np.sum(M@p)/np.sum(M)

    return p-pmean # P is (N+1)^2 column vector 

def correct_vhat_with_pressure(N,dt,vhat,p,Dx,Dy):
    '''
    Corrects v_hat using the pressure gradient.
    N: int, polynomial order
    dt: float, timestep
    vhat: (2, (N+1)^2) array, velocity coefficients
    p: ((N+1)^2,) array, pressure coefficients
    Dx, Dy: ((N+1)^2, (N+1)^2) arrays, differentiation tensors
    Returns vhathat: (2,(N+1)^2) array, corrected velocity coefficients
    '''
    gradp = np.zeros((2,(N+1)**2))
    gradp[0,:] = Dx@p
    gradp[1,:] = Dy@p
    vhathat = vhat - dt*gradp
    return vhathat

def helmholtz_update(N, dt, k, diffusivity, A, M, vhathat, vel_bound):
    '''
    Updates velocity coefficients by solving the Helmholtz equation.
    dt: float, timestep
    k: int, time integration order
    diffusivity: float, diffusion coefficient
    A: ((N+1)^2, (N+1)^2) array, stiffness matrix
    M: ((N+1)^2, (N+1)^2) array, mass matrix
    vhathat: (2, (N+1)^2) array, corrected velocity coefficients
    Returns sol_next: (2,(N+1)^2) array, updated velocity coefficients
    '''
    implicit_coeff = BDFk_coefs(k)[0] # Should be 11/6 for k=3?
    LHS = implicit_coeff*M - dt*diffusivity*A
    RHS_u = M@vhathat[0,:] # Chat suggests vhathat should also be multiplied by M
    RHS_v = M@vhathat[1,:] # Makes sense if Beta_k is multiplied by M
    LHS_mod, RHS_u_mod = modify_lhs_rhs_dirichlet(LHS,RHS_u,N,vel_bound[:,0]) # No need to modify both LHS!
    _, RHS_v_mod = modify_lhs_rhs_dirichlet(LHS,RHS_v,N,vel_bound[:,1])
    B = np.column_stack((RHS_u_mod, RHS_v_mod))   # shape ((n+1)^2, 2)
    sol_next = np.linalg.solve(LHS_mod, B)    # shape ((n+1)^2, 2) - apparantly numpy can solve 2 at once?
    return sol_next.T # shape (2, (n+1)^2)

def curlcurl(velocities,Dx,Dy):
    '''
    Computes curl(curl(velocity)) for a velocity field.
    velocities: (2, (N+1)^2) array, velocity coefficients [u, v]
    Dx, Dy: ((N+1)^2, (N+1)^2) arrays, differentiation tensors
    Returns: (2, (N+1)^2) array, curl(curl(velocity))
    '''
    [u,v] = velocities

    vxy = Dx@Dy@v
    vxx = Dx@Dx@v
    uxy = Dx@Dy@u
    uyy = Dy@Dy@u

    return np.array([vxy-uyy,uxy-vxx])


def evaluate_cfl(N,dt,vel):
    '''
    Compute rough estimate of CFL number given a polynomial number, timestep, and velocity array
    N: polynomial order
    dt: timestep
    vel: 2x(N+1)^2 velocity at current timestep
    '''
    gll_pts, _ = gll_pts_wts(N)
    max_vel = np.max(vel)
    min_dist = np.abs(gll_pts[1]-gll_pts[0])
    cfl = (max_vel*dt)/min_dist
    return cfl