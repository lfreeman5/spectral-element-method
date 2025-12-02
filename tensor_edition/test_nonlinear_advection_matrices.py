import numpy as np
from gll_utils import gll_pts_wts
from tensor_sem_utils import map_MM_to_M2M2, map_2d_to_1d, create_Jhat, create_C, nonlinear_advection_at_previous_time, nonlinear_advection_at_previous_time_textbook

if __name__ == "__main__":
    N = 10
    M = 10
    Npts, _ = gll_pts_wts(N)


    ux_coefs = np.random.random((N+1)*(N+1))
    uy_coefs = np.random.random((N+1)*(N+1))
    # print("ux_coefs = ", ux_coefs)

    Cvx, Cvy = nonlinear_advection_at_previous_time(N, M, ux_coefs, uy_coefs)
    Cvx_textbook, Cvy_textbook = nonlinear_advection_at_previous_time_textbook(N, ux_coefs, uy_coefs)

    # print("Cvx          = ", Cvx)
    # print("Cvx_textbook = ", Cvx_textbook)
    # print("Cvy = ", Cvy)

    print(f'Norm Cvx between notes and textbook: {np.linalg.norm(Cvx - Cvx_textbook)}')
    print(f'Norm Cvy between notes and textbook: {np.linalg.norm(Cvy - Cvy_textbook)}')