import numpy as np
import sympy as sp
from advection_diffusion_2d import transient_solution

if __name__ == "__main__":
    # Set up exact solution with homogenous dirichlet boundaries, compute forcing function
    x,y,t = sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('t')
    u_exact = sp.sin(sp.pi*t)*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    alpha = 1
    c = [-y,x]

    f = sp.diff(u_exact, 't') + c[0]*sp.diff(u_exact,'x') + c[1]*sp.diff(u_exact, 'y') - alpha * (sp.diff(u_exact, 'x', 2) + sp.diff(u_exact, 'y', 2))
    print(sp.simplify(f))