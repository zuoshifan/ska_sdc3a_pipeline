"""Compute score of the estimated power spectrum, the higer score the better result."""

import numpy as np


true_pk2d = np.loadtxt('/home/s1_tianlai/kfyu/SDC3/TestDataset/TestDatasetTRUTH_166MHz-181MHz.data')
test_pk2d = np.loadtxt('./TianlaiTest_166MHz-181MHz.data')
test_err = np.loadtxt('./TianlaiTest_166MHz-181MHz_errors.data')


def log_prob(pk_true, pk_estimate, pk_error):
    prob = 1.0 / (np.sqrt(2*np.pi) * pk_error) * np.exp(-(pk_true - pk_estimate)**2 / (2 * pk_error**2))
    return np.sum(np.log10(prob))

score = log_prob(true_pk2d, test_pk2d, test_err)
print('score: ', score)
