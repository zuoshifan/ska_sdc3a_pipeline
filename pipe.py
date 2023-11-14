import configparser
import numpy as np
from scipy import linalg as la
import h5py
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from radio_beam import Beam
from pca_sub import pca
# from osvd_sub import osvd
from osvd_sub_multiprocessing import osvd
from power_spectrum import power_spectrum_2d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read('config.ini')

# read in image data
img_name = config['DEFAULT']['img_file']

hdul = fits.open(img_name)
print(hdul.info())
# print(hdul[0].header)
data = hdul[0].data
print(data.shape, data.dtype)
nfreq, x, y = data.shape

# crop to reserve only central 16 arcsec * 900 / 3600 = 4 deg
N = 900//2
data = data[:, x//2-N:x//2+N, y//2-N:y//2+N]
nfreq, nra, ndec = data.shape

# read in necessary info from header
pix_size = hdul[0].header['CDELT2'] # deg
freq0 = hdul[0].header['CRVAL3']
freq0i = hdul[0].header['CRPIX3']
dfreq = hdul[0].header['CDELT3']
freq = np.linspace(freq0, freq0 + dfreq * (nfreq - freq0i), nfreq) # Hz
print(freq[0]*1.0e-6, freq[-1]*1.0e-6) # Mhz

# frequency bin (6 bins)
freq_bins = np.loadtxt('bins_frequency.txt').astype(int)
nfb = freq_bins.shape[0] # number of frequency bins
fi_bins = (nfreq//nfb) * np.arange(nfb+1)
fi_bins[-1] += 1
datas = [ data[fi_bins[i]:fi_bins[i+1]] for i in range(nfb) ]
freqs = [ freq[fi_bins[i]:fi_bins[i+1]] for i in range(nfb) ]
# print([ freq.shape for freq in freqs ])

# k bins
kper_mid = np.loadtxt('bins_kper.txt')
kpar_mid = np.loadtxt('bins_kpar.txt')
dkper = kper_mid[1] - kper_mid[0]
kper = np.concatenate([[kper_mid[0] - 0.5*dkper], kper_mid + 0.5*dkper])
dkpar = kpar_mid[1] - kpar_mid[0]
kpar = np.concatenate([[kpar_mid[0] - 0.5*dkpar], kpar_mid + 0.5*dkpar])

# read in station beam
beam_name = config['DEFAULT']['beam_file']

hdul = fits.open(beam_name)
print(hdul.info())
# print(hdul[0].header)
beam = hdul[0].data
print(beam.shape, beam.dtype)
nfreq, bx, by = beam.shape
beam = beam[:, bx//2-N:bx//2+N, by//2-N:by//2+N]
beams = [ beam[fi_bins[i]:fi_bins[i+1]] for i in range(nfb) ]
# print([ beam.shape for beam in beams ])
# print(beam.shape, beam.min(), beam.max())

# # correct station beam
# datas = [ data/beam for (data, beam) in zip(datas, beams) ]

# correct for common station beam (the lowest freq beam)
# datas = [ data/beam[0:1, :, :] for (data, beam) in zip(datas, beams) ]

# Cosmology model for unit conversion
# convert (MHz, deg, deg) to comoving (Mpc, Mpc, Mpc)
H0 = 100.0 # Hubble constant at z = 0, [km/sec/Mpc]
Om0 = 0.30964 # Omega matter: density of non-relativistic matter in units of the critical density at z = 0
cos = FlatLambdaCDM(H0, Om0)


for (fbi, data, freq) in zip(np.arange(nfb), datas, freqs):

    # Convert the image data from Jy/beam to K
    # the beam
    beam = Beam.from_fits_header(fits.getheader(img_name))
    # print(beam)
    image_data_K = (data * u.Jy).to(u.K, u.brightness_temperature(freq[:, np.newaxis, np.newaxis]*u.Hz, beam))

    data = image_data_K.value

    # subtract mean of data
    # data -= np.mean(data, axis=(1, 2))[:, np.newaxis, np.newaxis]


    # foreground subtraction
    # # PCA method
    # # 30 modes
    # nmode = 30
    # R = pca(data, nmode)
    # print(R.min(), R.max())

    # OSVD method
    nmode = 15000
    R = osvd(data, nmode)
    print(R.min(), R.max())


    # compute 2d power spectrum for each frequency bin
    # nu0 = freq[len(freq)//2] # central frequency
    nu0 = freq[-1] # end frequency
    nu_HI = 1420.406 # MHz
    z = 1.0e6 * nu_HI / nu0 - 1
    print(z)
    rz = cos.comoving_distance(1.0e6 * nu_HI / freq[0] - 1) - cos.comoving_distance(1.0e6 * nu_HI / freq[-1] - 1) # Comoving line-of-sight distance in Mpc
    rx = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    ry = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    print(rx, ry, rz)


    R = R.transpose(1, 2, 0)
    window = None
    # window = 'blackmanharris'
    # window = 'tukey'
    ps, err, kper_mid, kpar_mid, n_modes = power_spectrum_2d(R, kbins=[kper, kpar], binning=None, box_dims=[rx.value, ry.value, rz.value], return_modes=True, window=window)

    # save ps to file
    fl = f'Tianlai_{freq_bins[fbi][0]}MHz-{freq_bins[fbi][1]}MHz.data'
    np.savetxt(fl, ps.T, fmt='%.12f') # Note: transpose to make each row is of constant k_parallel

    # save 1sigma error to file
    fl = f'Tianlai_{freq_bins[fbi][0]}MHz-{freq_bins[fbi][1]}MHz_errors.data'
    np.savetxt(fl, err, fmt='%.12f')