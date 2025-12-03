import numpy as np
from gll_utils import gll_pts_wts
from tensor_sem_utils import create_Jhat, create_Bhat_Dhat, create_Dtilde
from ns_utils import  calc_v_hat

if __name__ == "__main__":
    N = 10
    M = 15
    Npts, _ = gll_pts_wts(N)

    J_hat = create_Jhat(N,M) 
    B_hat_M, D_hat = create_Bhat_Dhat(M)
    D_tilde = create_Dtilde(N,M)
    B_M = np.kron(B_hat_M, B_hat_M)

    dt = .001
    k = 3
    saved_vx = []
    saved_vy = []
    for i in range(k):
        saved_vx.append(np.random.random((N+1)*(N+1)))
        saved_vy.append(np.random.random((N+1)*(N+1)))


    v_hat_u, v_hat_v = calc_v_hat(k, dt, saved_vx, saved_vy, J_hat, B_M, D_tilde)
    print("v_hat_u = ", v_hat_u[:5])
    print("v_hat_v = ", v_hat_v[:5])