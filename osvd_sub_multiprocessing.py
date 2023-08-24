import numpy as np
from scipy import linalg as la
import multiprocessing


# Define a helper function that computes Ut, St and Vt for a given i
def compute_USV(i, s3, V3T):
    # print(f'Computing {i}...')
    U, s, VT = la.svd(V3T, full_matrices=True)
    # U: (nx, nx), s: (ns1,), VT: (ny, ny)
    Ut_i = U
    Vt_i = VT
    St_i = np.zeros_like(V3T)
    for j in range(len(s)):
        St_i[j, j] = s3 * s[j]
    return Ut_i, St_i, Vt_i

def osvd_decomp(A):
    nx, ny, nz = A.shape
    A3 = A.reshape(nx*ny, nz).T
    U3, s3, V3T = la.svd(A3, full_matrices=False)
    # U3: (nz, ns), s3: (ns, ), V3T: (ns, nx*ny)

    ns = len(s3)
    Ut = np.zeros((nx, nx, ns), dtype=A.dtype)
    St = np.zeros((nx, ny, ns), dtype=A.dtype)
    Vt = np.zeros((ny, ny, ns), dtype=A.dtype)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(10)

    # Use pool.map to apply the helper function to each i in parallel
    USV_list = pool.starmap(compute_USV, zip(range(ns), [s3[i] for i in range(ns)], [V3T[i, :].reshape(nx, ny) for i in range(ns)]))

    # Assign the results to the corresponding slices of Ut, St and Vt
    for i in range(ns):
        Ut[:, :, i] = USV_list[i][0]
        St[:, :, i] = USV_list[i][1]
        Vt[:, :, i] = USV_list[i][2]

    return Ut, St, Vt, U3


# Define a helper function that computes R for a given k
def compute_R(k, U, S, V):
    # print(f'Computing {k}...')
    R_k = U @ S @ V
    return R_k[:, :, np.newaxis]

def osvd(data, nmode=20000):
    # foreground subtraction
    # OSVD method
    data = data.transpose(1, 2, 0)
    nx, ny, nf = data.shape
    U, S, V, U3 = osvd_decomp(data)

    # # save osvd result
    # import h5py
    # with h5py.File('msw_image_osvd_decomp_ZW3_c4deg_fbi4.hdf5', 'w') as f:
    #     f.create_dataset('U', data=U)
    #     f.create_dataset('S', data=S)
    #     f.create_dataset('V', data=V)
    #     f.create_dataset('U3', data=U3)

    # # read in data
    # import h5py
    # with h5py.File('msw_image_osvd_decomp_ZW3_c4deg_fbi4.hdf5', 'r') as f:
    #     U = f['U'][:]
    #     S = f['S'][:]
    #     V = f['V'][:]
    #     U3 = f['U3'][:]


    s = np.sort(S, axis=None)
    th = s[-nmode]
    S[S>th] = 0.0

    # # plot s
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.semilogy(s[s>0][::-1][:50000], 'ro')
    # plt.savefig('osvd_eigvals_50000.png')
    # plt.close()

    # Create a pool of worker processes
    pool = multiprocessing.Pool(10)

    # Use pool.map to apply the helper function to each k in parallel
    R_list = pool.starmap(compute_R, zip(range(nf), [U[:, :, k] for k in range(nf)], [S[:, :, k] for k in range(nf)], [V[:, :, k] for k in range(nf)]))

    # Concatenate the results into a single array
    R = np.concatenate(R_list, axis=2)
    R = (R.reshape((nx*ny, nf)) @ U3.T).reshape((nx, ny, nf))

    return R.transpose(2, 0, 1)