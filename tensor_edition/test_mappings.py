import numpy as np
from lagrange_utils import construct_solution_2d_fast
from tensor_sem_utils import map_2d_to_1d, map_1d_to_2d
N=4
# create labeled boundary field as above
A = np.zeros((N+1,N+1), dtype=float)
A[0,:] = 10 
A[-1,:] = 20 
A[:,0] = 30 
A[:,-1] = 40
v = map_2d_to_1d(A, N)
# reconstruct function
u_func = construct_solution_2d_fast(A, np.linspace(-1,1,N+1))
# evaluate at the GLL pts used by plotting (len(x)=len(y)=N+1)
x = np.linspace(-1,1,N+1)
y = np.linspace(-1,1,N+1)
U = u_func(x,y)   # shape (len(x), len(y))
print("U.shape:", U.shape)
print("U matrix (rows correspond to x-index, cols to y-index):\n", U)
print("Transposed U.T:\n", U.T)