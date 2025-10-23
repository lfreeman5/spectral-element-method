from sem_utils import construct_ax_matrix_2d, construct_ay_matrix_2d, construct_m_matrix_2d, construct_load_matrix_2d
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

