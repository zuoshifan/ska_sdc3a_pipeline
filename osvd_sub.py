import numpy as np
from scipy import linalg as la


def osvd_decomp(A):
    nx, ny, nz = A.shape
    A3 = A.reshape(nx*ny, nz).T
    U3, s3, V3T = la.svd(A3, full_matrices=False)
    # U3: (nz, ns), s3: (ns, ), V3T: (ns, nx*ny)

    ns = len(s3)
    Ut = np.zeros((nx, nx, ns), dtype=A.dtype)
    St = np.zeros((nx, ny, ns), dtype=A.dtype)
    Vt = np.zeros((ny, ny, ns), dtype=A.dtype)

    for i in range(ns):
        U, s, VT = la.svd(V3T[i, :].reshape(nx, ny), full_matrices=True)
        # U: (nx, nx), s: (ns1,), VT: (ny, ny)
        Ut[:, :, i] = U
        Vt[:, :, i] = VT
        for j in range(len(s)):
            St[j, j, i] = s3[i] * s[j]

    return Ut, St, Vt, U3


def osvd(data, nmode=20000):
    # foreground subtraction
    # OSVD method
    data = data.transpose(1, 2, 0)
    nx, ny, nf = data.shape
    U, S, V, U3 = osvd_decomp(data)

    s = np.sort(S, axis=None)
    th = s[-nmode]
    S[S>th] = 0.0

    R = np.zeros_like(data)
    for k in range(nf):
        R[:, :, k] = U[:, :, k] @ S[:, :, k] @ V[:, :, k]
    R = (R.reshape((nx*ny, nf)) @ U3.T).reshape((nx, ny, nf))

    return R.transpose(2, 0, 1)