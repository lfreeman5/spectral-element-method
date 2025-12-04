from tensor_sem_utils import map_1d_to_2d, map_2d_to_1d
import numpy as np
N=4
# build a flat vector of zeros then set the left/right indices using the same formulas
vec = np.zeros((N+1)**2, dtype=float)
# left, right
for j in range(N+1):
    kL = (N+1)*j
    kR = (N+1)*j + N
    vec[kL] = 100+j
    vec[kR] = 200+j
# bottom/top
for i in range(N+1):
    kB = i
    kT = i+(N+1)*N
    vec[kB] = 300+i
    vec[kT] = 400+i
A = map_1d_to_2d(vec, N)
print("A 2D from vec:\n", A)
# show boundaries:
print("left column (i=0):", A[0,:])
print("right column (i=N):", A[-1,:])
print("bottom row (j=0):", A[:,0])
print("top row (j=N):", A[:,-1])