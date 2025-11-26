import numpy as np
from gll_utils import gll_pts_wts
from tensor_sem_utils import map_MM_to_M2M2, map_2d_to_1d, create_Jhat, create_C

if __name__ == "__main__":
    N = 10
    M = 15
    Npts, _ = gll_pts_wts(N)
    Mpts, _ = gll_pts_wts(M)

    # Create velocity field
    Cx = lambda x,y: -y
    Cy = lambda x,y: x

    # Compute velocity field on N GLL points
    CxNN = np.zeros(((N+1),(N+1)))
    CyNN = np.zeros(((N+1),(N+1)))
    for i in range(N+1):
        for j in range(N+1):
            CxNN[i,j] = Cx(Npts[i], Npts[j])
            CyNN[i,j] = Cy(Npts[i], Npts[j])
    CxN2 = map_2d_to_1d(CxNN, N)
    CyN2 = map_2d_to_1d(CyNN, N)

    # Interpolate N velocity field to M velocity field
    JhatM = create_Jhat(N,M)
    CxM2_interp = (np.kron(JhatM,JhatM))@CxN2
    CyM2_interp = (np.kron(JhatM,JhatM))@CyN2

    # Compute velocity field directly on M GLL points
    CxMM = np.zeros(((M+1),(M+1)))
    CyMM = np.zeros(((M+1),(M+1)))
    for i in range(M+1):
        for j in range(M+1):
            CxMM[i,j] = Cx(Mpts[i], Mpts[j])
            CyMM[i,j] = Cy(Mpts[i], Mpts[j])
    CxM2_direct = map_2d_to_1d(CxMM, M)
    CyM2_direct = map_2d_to_1d(CyMM, M)


    # Compare:
    print(f'Norm on CXM2 between interpolated and direct: {np.linalg.norm(CxM2_interp-CxM2_direct)}')
    print(f'Norm on CYM2 between interpolated and direct: {np.linalg.norm(CyM2_interp-CyM2_direct)}')

    # Compute advection operator interpolated and direct
    C_interp = create_C(N,M,CxM2_interp,CyM2_interp)
    C_direct = create_C(N,M,CxM2_direct,CyM2_direct)
    print(f'\nNorm of difference between interpolated and direct advection operator: {np.linalg.norm(C_interp-C_direct)}')