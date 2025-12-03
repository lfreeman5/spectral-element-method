import numpy as np
from tensor_sem_utils import *
from fractions import Fraction


def calc_v_hat(k, dt, saved_u_coefs, saved_v_coefs, J_hat, B_M, D_tilde):
    '''
    based on eq. 6.5.8 in the text
    calculates v_hat used in pressure poisson solve.
    k is order of BDF (implicit) terms and Adams Bashforth (explicit) terms
    saved_vx_coefs and saved_vy_coefs are arrays with k columns being [n data, n-1 data, n-2 data, ...]
    '''
    # get integration coefficients
    bj = AB_coefs(k)
    __, beta_k_minus_j = BDFk_coefs(k)

    v_hat_u = np.zeros(len(saved_u_coefs[-1]))
    v_hat_v = np.zeros(len(saved_v_coefs[-1])) 
    for j in range(k):
    #     print("      bj = ", Fraction(bj[j]).limit_denominator())
    #     print("beta k-j = ", Fraction(beta_k_minus_j[j]).limit_denominator())
        # -(1+j) because we want most recent data (list element -1), then the one before that (list element -2), and so on


        Cu, Cv = nonlinear_advection_at_previous_time(saved_u_coefs[j,:], saved_v_coefs[j,:], J_hat, B_M, D_tilde) # j = 0 is
        v_hat_u += -beta_k_minus_j[j]*B_M@saved_u_coefs[j,:] + dt*bj[j]*Cu
        v_hat_v += -beta_k_minus_j[j]*B_M@saved_v_coefs[j,:] + dt*bj[j]*Cv

    return np.array([v_hat_u, v_hat_v]).T # Change shape to be (N+1)^2 x 2

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
    vel_boundary: (4, 2) array, velocity boundary conditions
    Returns p: ((N+1)^2,) array, pressure coefficients
    '''

    # Set up boundary term 
    B = np.zeros(((N+1)^2,(N+1)^2))

    # GLL pts, wts
    _,wts = gll_pts_wts(N)

    # Time integration coeffs
    expl_coeffs = AB_coefs(k)
    impl_coeffs = BDFk_coefs(k)

    # Compute div(vhat)/dt
    div_vhat = Dx@vhat[0]+Dy@vhat[1]

    # Compute curl(curl(v^{n+1-j})) for non-boundary-dependent part
    inner_boundary_term = np.zeros_like(vhat)
    for j in range(k):
        inner_boundary_term += expl_coeffs[j]*curlcurl(vel[-(j+1)],Dx,Dy) #
    inner_boundary_term -= vhat/dt
    inner_boundary_term = map_1d_to_2d(inner_boundary_term, N)
    # At this point need to move to side-by-side based on velocity in
    # Bottom: 
    bottom_normal = np.array([0,-1])
    bottom_term = inner_boundary_term[:,0] + np.tile(vel_boundary[0], (N+1,1))/dt # Last part is the g/dt component
    bottom_dotted = -np.dot(bottom_term, bottom_normal) # This should be an (N+1)x1 column vector???
    B[:,0] += bottom_dotted*wts
    # Right: 
    right_normal = np.array([1,0])
    right_term = inner_boundary_term[-1,:] + np.tile(vel_boundary[3], (N+1,1))/dt
    right_dotted = -np.dot(right_term, right_normal)
    B[-1,:] += right_dotted*wts
    # Top: 
    top_normal = np.array([0,1])
    top_term = inner_boundary_term[:,-1] + np.tile(vel_boundary[1], (N+1,1))/dt
    top_dotted = -np.dot(top_term, top_normal)
    B[:,-1] += -top_dotted*wts
    # Left: 
    left_normal = np.array([-1,0])
    left_term = inner_boundary_term[0,:] + np.tile(vel_boundary[2], (N+1,1))/dt
    left_dotted = -np.dot(left_term,left_normal)
    B[0,:] += -left_dotted*wts
    # Then assemble the boundary condition with constant g. Then map full thing 1d-to-2d and extract the relevant side
    # ^^^ That is horribly inefficient

    # Then we move to the pressure solve
    # Solve (A,b) where A is stiffness, b=f_ij+boundary
    # assemble f_ij based on div vhat, I think M kron div vhat? 
    F = M@div_vhat
    LHS = A # Check sign on this - should incorporate the negative in constructing it
    RHS = F + map_2d_to_1d(B,N)
    p = np.linalg.solve(LHS, RHS)

    return p # P is (N+1)^2 column vector 

def correct_vhat_with_pressure(N,dt,vhat,p,Dx,Dy):
    '''
    Corrects v_hat using the pressure gradient.
    N: int, polynomial order
    dt: float, timestep
    vhat: ((N+1)^2, 2) array, velocity coefficients
    p: ((N+1)^2,) array, pressure coefficients
    Dx, Dy: ((N+1)^2, (N+1)^2) arrays, differentiation tensors
    Returns vhathat: ((N+1)^2, 2) array, corrected velocity coefficients
    '''
    gradp = np.zeros(((N+1)^2,2))
    gradp[:,0] = Dx@p
    gradp[:,1] = Dy@p
    vhathat = vhat - dt*gradp
    return vhathat

def helmholtz_update(dt, k, diffusivity, A, M, vhathat):
    '''
    Updates velocity coefficients by solving the Helmholtz equation.
    dt: float, timestep
    k: int, time integration order
    diffusivity: float, diffusion coefficient
    A: ((N+1)^2, (N+1)^2) array, stiffness matrix
    M: ((N+1)^2, (N+1)^2) array, mass matrix
    vhathat: ((N+1)^2, 2) array, corrected velocity coefficients
    Returns sol_next: ((N+1)^2, 2) array, updated velocity coefficients
    '''
    implicit_coeff = BDFk_coefs(k)[0] # Should be 11/6 for k=3?
    LHS = implicit_coeff*M - dt*diffusivity*A
    RHS_u = vhathat[:,0]
    RHS_v = vhathat[:,1]
    B = np.column_stack((RHS_u, RHS_v))   # shape (n, 2)
    sol_next = np.linalg.solve(LHS, B)    # shape (n, 2) - apparantly numpy can solve 2 at once?
    return sol_next

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

    return np.array([vxy-uyy,vxx-uxy])
