import numpy as np
from scipy import linalg as la


def pca(data, nmode=10):
    # foreground subtraction
    # PCA method
    nf, nra, ndec = data.shape
    D = data.reshape(nf, -1)
    C = np.dot(D, D.T) / (nra * ndec)
    e, U = la.eigh(C)

    # # plot eigvals
    # plt.figure()
    # plt.semilogy(e[::-1], 'ro')
    # plt.savefig(f'eigvals_freq_bin_{fbi}.png')
    # plt.close()

    # 10 modes
    s = np.zeros_like(e)
    s[-nmode:] = 1.0
    F = np.dot(np.dot(U*s, U.T), D)
    F = F.reshape((nf, nra, ndec))
    R = data - F # residual 21 cm signal + noise

    return R