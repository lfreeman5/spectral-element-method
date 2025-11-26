import numpy as np

def eval_pn(x,N):
    '''
    Evaluates the Nth legendre polynomial at x where x ∈ [-1,1]
    Supports vectorized X
    Note that eventually numpy supports evaluating all up to order N simultaneously, which is more efficient
    '''
    assert (x>=-1 and x<=1), 'Legendre polynomials only defined on x ∈ [-1,1]'
    c = np.zeros(N+1)
    c[-1] = 1
    return np.polynomial.legendre.legval(x, c)

def gll_pts_wts(N):
    '''
    Get GLL weights and points for integration with Nth legendre polynomial
    The points are the roots of P'_{n-1}, ie the derivative of the N-1 polynomial
       as well as -1 and 1
    Returns tuple:
        (pts, wts) where both are 1x(N+1) numpy arrays  
    '''
    from numpy.polynomial.legendre import legder, legroots, legval # legder gets derivative coefficients, legroots gets roots of polynomial with given coeffs, legval evaluates polynomial at points
    if N < 1:
        raise ValueError("Order N must be >= 1")
    c=np.zeros(N+1) # coefficients of P_N, initially there are N+1 coefficients all 0
    c[-1]=1 # set coefficient of x^N to 1 because we want P_N
    dC = legder(c) # coefficients of P'_N. legder takes the derivative of the polynomial whose coefficients are in c
    pts = np.concatenate(([-1.],legroots(dC),[1.])) # GLL points are roots of P'_{N} plus -1 and 1
    wts = 2.0 / (N*(N+1) * legval(pts, c)**2)
    return pts, wts

def integrate_gll(a,b,f,N):
    '''
    Performs integral of f(x) from a to b using GLL
    Uses polynomial of order N
    '''
    pts, wts = gll_pts_wts(N)
    x_pts = (b-a)/2*pts + (b+a)/2
    f_evals = [f(x_val) for x_val in x_pts]
    return (b-a)/2 * np.dot(wts, f_evals)

def print_matrix(A, fmt="{:10.4f}"):
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        if A.shape[0] == 1:  # row vector
            print(" ".join(fmt.format(x) for x in A))
        else:  # column vector
            for x in A:
                print(fmt.format(x))
    else:
        for row in A:
            print(" ".join(fmt.format(x) for x in row))