import numpy as np

'''
This file contains some functions for working with Lagrange polynomials in GLL space
'''

def create_lagrange_poly(i,x_arr):
    ''''
    returns a function L_i, the ith lagrange polynomial
    0<=i<=N where N is the GLL P_N order.
    x_arr is an array of the GLL points on [-1,1] with size N+1
    NOTE: there are some big speedups left on the table, for now I wanted it to be simple code. Chat has some vectorizations which give ~80x speedup
    '''
    N = len(x_arr)-1
    def L_i(x):
        value = 1
        for j in range(0,N+1):
            if(j==i): continue
            value *= (x-x_arr[j])/(x_arr[i]-x_arr[j])
        return value
    return L_i