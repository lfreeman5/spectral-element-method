import numpy as np
from gll_utils import gll_pts_wts
from ns_utils import  calc_v_hat

if __name__ == "__main__":
    N = 10
    M = 15
    Npts, _ = gll_pts_wts(N)

    dt = .001
    k = 3
    saved_vx = []
    saved_vy = []
    for i in range(k):
        saved_vx.append(np.random.random((N+1)*(N+1)))
        saved_vy.append(np.random.random((N+1)*(N+1)))


    v_hat_x, v_hat_y = calc_v_hat(N, M, k, dt, saved_vx, saved_vy)
    print("v_hat_x = ", v_hat_x[:5])
    print("v_hat_y = ", v_hat_y[:5])