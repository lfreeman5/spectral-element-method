from sem_utils import *
import time
import numpy as np

if __name__ == "__main__":
    Ns = [2, 4, 8, 16, 32, 64]

    for N in Ns:
        t0 = time.time()
        Ax = construct_m_matrix_2d(N)
        t1 = time.time()
        size = Ax.nbytes if hasattr(Ax, 'nbytes') else Ax.__sizeof__()
        print(f"M N={N}, time={t1-t0:.6f}s, size={size} bytes")
    print('\n\n')

    for N in Ns:
        f = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(x*y)
        t0 = time.time()
        Ax = construct_load_matrix_2d(N,f)
        t1 = time.time()
        size = Ax.nbytes if hasattr(Ax, 'nbytes') else Ax.__sizeof__()
        print(f"F N={N}, time={t1-t0:.6f}s, size={size} bytes")
    print('\n\n')

    # Test mapping
    print(f'Testing 2D-->1D-->2D mapping')
    mapped = map_2d_to_1d(Ax,Ns[-1])
    unmapped = map_1d_to_2d(mapped,Ns[-1])
    print(f'Difference between original and mapped-->unmapped: {np.linalg.norm(unmapped-Ax)}\n\n')

    for N in Ns:
        t0 = time.time()
        Ax = construct_ax_matrix_2d(N)
        t1 = time.time()
        size = Ax.nbytes if hasattr(Ax, 'nbytes') else Ax.__sizeof__()
        print(f"Ax N={N}, time={t1-t0:.6f}s, size={size} bytes")
    print('\n\n')

    for N in Ns:
        t0 = time.time()
        Ax = construct_ay_matrix_2d(N)
        t1 = time.time()
        size = Ax.nbytes if hasattr(Ax, 'nbytes') else Ax.__sizeof__()
        print(f"Ay N={N}, time={t1-t0:.6f}s, size={size} bytes")

