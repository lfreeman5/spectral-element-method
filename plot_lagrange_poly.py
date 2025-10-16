import numpy as np
import matplotlib.pyplot as plt
from gll_utils import gll_pts_wts
from lagrange_utils import create_lagrange_poly, create_lagrange_derivative, create_lagrange_derivative_continuous, create_lagrange_derivative_unified

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
    plt.title(f'Lagrange basis polynomials on GLL nodes (N={N})')

    for k in range(N+1):
        plt.figure(figsize=(8,5))
        Lp_cont = create_lagrange_derivative_continuous(k, pts)
        Lp_disc = create_lagrange_derivative(k, pts)
        Lp_unified = create_lagrange_derivative_unified(k, pts)
        for n in range(N+1):
            print(f'discrete vs unified error: {Lp_disc(pts[n])-Lp_unified(pts[n])}')
        plt.plot(x_space, [Lp_cont(x) for x in x_space], label=f"$L'_{k}$ (continuous)")
        plt.scatter(pts, [Lp_disc(pts[j]) for j in range(N+1)], label=f"$L'_{k}$ (discrete at GLL nodes)")
        plt.plot(x_space, [Lp_unified(x) for x in x_space], label=f"$L'_{k}$ (unified)", linestyle=':', color='green')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.title(f"Derivative of L_{k}: continuous vs discrete vs unified (N={N})")

        plt.show()
