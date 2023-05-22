import numpy as np
from scipy import linalg as la
# import h5py
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from radio_beam import Beam
from power_spectrum import power_spectrum_2d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# read in image data
filename = '/home/s1_tianlai/SKA/SDC3/ZW3.msw_image.fits'

hdul = fits.open(filename)
print(hdul.info())
# print(hdul[0].header)
data = hdul[0].data
print(data.shape, data.dtype)
# crop to reserve only central 1024 x 1024
data = data[:, 1024-512:1024+512, 1024-512:1024+512]
nfreq, nra, ndec = data.shape

# read in necessary info from header
pix_size = hdul[0].header['CDELT2'] # deg
freq0 = hdul[0].header['CRVAL3']
freq0i = hdul[0].header['CRPIX3']
dfreq = hdul[0].header['CDELT3']
freq = np.linspace(freq0, freq0 + dfreq * (nfreq - freq0i), nfreq) # Hz
print(freq[0]*1.0e-6, freq[-1]*1.0e-6) # Mhz


# foreground subtraction
# PCA method
D = data.reshape(nfreq, -1)
C = np.dot(D, D.T) / (nra * ndec)
e, U = la.eigh(C)

# 30 modes
nmode = 30
s = np.zeros_like(e)
s[-nmode:] = 1.0
F = np.dot(np.dot(U*s, U.T), D)
F = F.reshape((nfreq, nra, ndec))
R = data - F # residual 21 cm signal + noise
R = R.reshape((nfreq, nra, ndec))


# # Convert the image data from Jy/beam to K
# the beam
beam = Beam.from_fits_header(fits.getheader(filename))
# print(beam)
image_data_K = (R * u.Jy).to(u.K, u.brightness_temperature(freq[:, np.newaxis, np.newaxis]*u.Hz, beam))

R = image_data_K.value
# print(data)


# unit conversion
# convert (MHz, deg, deg) to comoving (Mpc, Mpc, Mpc)
H0 = 100.0 # Hubble constant at z = 0, [km/sec/Mpc]
Om0 = 0.30964 # Omega matter: density of non-relativistic matter in units of the critical density at z = 0
cos = FlatLambdaCDM(H0, Om0)


# frequency bin (6 bins)
freq_bins = np.loadtxt('bins_frequency.txt').astype(int)
nfb = freq_bins.shape[0] # number of frequency bins
Rs = [ R[i*(nfreq//nfb):(i+1)*(nfreq//nfb)] for i in range(nfb-1) ] + [ R[(nfb-1)*(nfreq//nfb):] ]
freqs = [ freq[i*(nfreq//nfb):(i+1)*(nfreq//nfb)] for i in range(nfb-1) ] + [ freq[(nfb-1)*(nfreq//nfb):] ]
# print([ freq.shape for freq in freqs ])

# k bins
kper_mid = np.loadtxt('bins_kper.txt')
kpar_mid = np.loadtxt('bins_kpar.txt')
dkper = kper_mid[1] - kper_mid[0]
kper = np.concatenate([[kper_mid[0] - 0.5*dkper], kper_mid + 0.5*dkper])
dkpar = kpar_mid[1] - kpar_mid[0]
kpar = np.concatenate([[kpar_mid[0] - 0.5*dkpar], kpar_mid + 0.5*dkpar])


# compute 2d power spectrum for each frequency bin
for fbi in range(nfb):

    freq = freqs[fbi]

    nu0 = freq[len(freq)//2] # central frequency
    nu_HI = 1420.406 # MHz
    z = 1.0e6 * nu_HI / nu0 - 1
    print(z)
    rz = cos.comoving_distance(1.0e6 * nu_HI / freq[0] - 1) - cos.comoving_distance(1.0e6 * nu_HI / freq[-1] - 1) # Comoving line-of-sight distance in Mpc
    rx = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    ry = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    print(rx, ry, rz)


    R = Rs[fbi].transpose(1, 2, 0)
    ps, kper_mid, kpar_mid, n_modes = power_spectrum_2d(R, kbins=[kper, kpar], binning=None, box_dims=[rx.value, ry.value, rz.value], return_modes=True)

    # save ps to file
    fl = f'Tianlai_{freq_bins[fbi][0]}MHz_{freq_bins[fbi][1]}MHz.data'
    np.savetxt(fl, ps.T, fmt='%.12f') # Note: transpose to make each row is of constant k_parallel

    # TODO:
    # to get 1sigma error of ps
    fl = f'Tianlai_{freq_bins[fbi][0]}MHz_{freq_bins[fbi][1]}MHz_errors.data'
    error = np.zeros_like(ps)
    np.savetxt(fl, error, fmt='%.12f')