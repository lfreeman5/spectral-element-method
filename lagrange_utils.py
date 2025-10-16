import numpy as np

'''
This file contains some functions for working with Lagrange polynomials in GLL space
'''

def create_lagrange_poly(i,x_arr):
    ''''
    returns a function L_i, the ith lagrange polynomial
    0<=i<=N where N is the GLL P_N order.
    x_arr is an array of the GLL points on [-1,1] with size N+1
    NOTE: there are some big speedups left on the table, for now I wanted it to be simple code. Chat gives some vectorizations which give ~80x speedup
    '''
    N = len(x_arr)-1
    def L_i(x):
        value = 1
        for j in range(0,N+1):
            if(j==i): continue
            value *= (x-x_arr[j])/(x_arr[i]-x_arr[j])
        return value
    return L_i

def create_lagrange_derivative(i,x_arr):
    '''
    returns a function l'_i, the derivative of the ith lagrange polynomial
    However, for now I only care about the derivative on the GLL nodes
    so, the returned function L_i_prime is only defined on GLL nodes x_j
    def L_i'(x_j):
    '''
    
    x_arr = np.asarray(x_arr, dtype=float)
    n = x_arr.size
    w = np.empty(n, dtype=float)
    for k in range(n):
        diffs = x_arr[k] - np.delete(x_arr, k)
        w[k] = 1.0 / np.prod(diffs)

    def L_i_prime(x_j):
        idx = np.where(np.isclose(x_arr, x_j, atol=1e-8))[0]
        assert idx.size > 0, 'create_lagrange_derivative is only defined for x=x_j'
        j = idx[0]
        if j == i:
            # l_i'(x_i) = sum_{m != i} 1 / (x_i - x_m)
            return float(np.sum(1.0 / (x_arr[i] - np.delete(x_arr, i))))
        else:
            # l_i'(x_j) = (w_i / w_j) / (x_j - x_i)
            return float((w[i] / w[j]) / (x_arr[j] - x_arr[i]))
    return L_i_prime

def create_lagrange_derivative_continuous(i,x_arr):
    '''
    not very performant. Mostly for plotting purposes.
    '''
    def L_i_prime(x):
        L_i_prime_pts = create_lagrange_derivative(i,x_arr)
        if(x in x_arr):
            return L_i_prime_pts(x)
        else:
            li = create_lagrange_poly(i,x_arr)
            x_m = np.array([x_arr[m] for m in range(0,len(x_arr)) if m != i])
            return li(x) * np.sum(1/(x-x_m))
    return L_i_prime

def create_lagrange_derivative_unified(i, x_arr):
    n = len(x_arr)
    denom = np.prod([x_arr[i]-x_arr[j] for j in range(n) if j!=i])
    def L_i_prime(x):
        return np.sum([np.prod([x-x_arr[k] for k in range(n) if k!= i and k!=j]) for j in range(n) if j!=i])/denom
    return L_i_prime

def construct_solution(u_arr, x_arr):
    '''
        Constructs the SEM solution u(x) ≈ sum(u_i*l_i(x))
    '''
    funcs = [create_lagrange_poly(i, x_arr) for i in range(len(x_arr))]
    def u_approx(x):
        return sum(u_arr[i] * funcs[i](x) for i in range(len(x_arr)))
    return u_approx
