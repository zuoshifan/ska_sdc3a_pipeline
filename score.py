"""Compute score of the estimated power spectrum, the higer score the better result."""

import configparser
import numpy as np


config = configparser.ConfigParser()
config.read('config.ini')

true_pk2d = np.loadtxt(config['DEFAULT']['true_pk2d_file'])
test_pk2d = np.loadtxt(config['DEFAULT']['test_pk2d_file'])
test_err = np.loadtxt(config['DEFAULT']['test_err_file'])


def log_prob(pk_true, pk_estimate, pk_error):
    prob = 1.0 / (np.sqrt(2*np.pi) * pk_error) * np.exp(-(pk_true - pk_estimate)**2 / (2 * pk_error**2))
    return np.sum(np.log10(prob))

score = log_prob(true_pk2d, test_pk2d, test_err)
print('score: ', score)
