import numpy as np
from tensor_sem_utils import *
from fractions import Fraction


def calc_v_hat(k, dt, saved_u_coefs, saved_v_coefs, J_hat, B_M, D_tilde):
    '''
    based on eq. 6.5.8 in the text
    calculates v_hat used in pressure poisson solve.
    k is order of BDF (implicit) terms and Adams Bashforth (explicit) terms
    saved_vx_coefs and saved_vy_coefs should be lists of np arrays with the most recent vx or vy appended to the end of the list
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
        Cu, Cv = nonlinear_advection_at_previous_time(saved_u_coefs[-(1+j)], saved_v_coefs[-(1+j)], J_hat, B_M, D_tilde) # -(1+j) because we want most recent
        v_hat_u += -beta_k_minus_j[j]*saved_u_coefs[-(1+j)] + dt*bj[j]*Cu
        v_hat_v += -beta_k_minus_j[j]*saved_v_coefs[-(1+j)] + dt*bj[j]*Cv

    return v_hat_u, v_hat_v