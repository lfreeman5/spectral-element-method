import numpy as np
import matplotlib.pyplot as plt
from gll_utils import gll_pts_wts
from lagrange_utils import create_lagrange_poly

if __name__ == '__main__':
    N = 8 # Order for P_N, the Legendre polynomial
    pts, _ = gll_pts_wts(N) # GLL points for P_N, len(pts)=N+1
    lagrange_polys = [create_lagrange_poly(i,pts) for i in range(0,N+1)]

    plt.figure(figsize=(8,5))
    x_space = np.linspace(-1,1,100+10*N)
    for i,poly in enumerate(lagrange_polys):
        plt.plot(x_space,poly(x_space),label=f'$L_{i}$')
        plt.axvline(x=pts[i],color='k',linestyle='--')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
