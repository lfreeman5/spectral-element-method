import numpy as np
from tensor_sem_utils import *
from fractions import Fraction


def calc_v_hat(N, M, k, dt, saved_vx_coefs, saved_vy_coefs):
    '''
    based on eq. 6.5.8 in the text
    calculates v_hat used in pressure poisson solve.
    k is order of BDF (implicit) terms and Adams Bashforth (explicit) terms
    saved_vx_coefs and saved_vy_coefs should be lists of np arrays with the most recent vx or vy appended to the end of the list
    '''
    # get integration coefficients
    bj = AB_coefs(k)
    __, beta_k_minus_j = BDFk_coefs(k)

    v_hat_x = np.zeros(len(saved_vx_coefs[-1]))
    v_hat_y = np.zeros(len(saved_vy_coefs[-1])) 
    for j in range(k):
    #     print("      bj = ", Fraction(bj[j]).limit_denominator())
    #     print("beta k-j = ", Fraction(beta_k_minus_j[j]).limit_denominator())
        # -(1+j) because we want most recent data (list element -1), then the one before that (list element -2), and so on
        Cvx, Cvy = nonlinear_advection_at_previous_time(N,M,saved_vx_coefs[-(1+j)], saved_vy_coefs[-(1+j)]) # -(1+j) because we want most recent
        v_hat_x += -beta_k_minus_j[j]*saved_vx_coefs[-(1+j)] + dt*bj[j]*Cvx
        v_hat_y += -beta_k_minus_j[j]*saved_vy_coefs[-(1+j)] + dt*bj[j]*Cvy

    return v_hat_x, v_hat_y