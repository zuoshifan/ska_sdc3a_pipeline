import numpy as np
import h5py
from astropy.io import fits
from astropy import units as u
# from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from radio_beam import Beam
from power_spectrum import power_spectrum_2d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# read in image data
img_name = '/home/s1_tianlai/kfyu/SDC3/TestDataset/TestDataset.msw_image.fits'

hdul = fits.open(img_name)
print(hdul.info())
# print(hdul[0].header)
data = hdul[0].data
print(data.shape, data.dtype)
# crop to reserve only central 1024 x 1024
# data = data[:, 1024-512:1024+512, 1024-512:1024+512]
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

# # read in station beam
# beam_name = '/home/s1_tianlai/SKA/SDC3/station_beam.fits'

# hdul = fits.open(beam_name)
# print(hdul.info())
# # print(hdul[0].header)
# beam = hdul[0].data
# print(beam.shape, beam.dtype)
# nfreq, bx, by = beam.shape
# beam = beam[:, bx//2-N:bx//2+N, by//2-N:by//2+N]
# # beams = [ beam[fi_bins[i]:fi_bins[i+1]] for i in range(nfb) ]

# # correct for common station beam (the lowest freq beam)
# # data = data/beam[0:1, :, :]
# # data = data/beam[600:601, :, :]


# Convert the image data from Jy/beam to K
# the beam
beam = Beam.from_fits_header(fits.getheader(img_name))
# print(beam)
image_data_K = (data * u.Jy).to(u.K, u.brightness_temperature(freq[:, np.newaxis, np.newaxis]*u.Hz, beam))

data = image_data_K.value

# subtract mean of data
# data -= np.mean(data, axis=(1, 2))[:, np.newaxis, np.newaxis]


# no foreground in TestDataset
R = data

# # save R to file for test
# with h5py.File('test_R.hdf5', 'w') as f:
#     f.create_dataset('R', data=R)
# exit()


# unit conversion
# convert (MHz, deg, deg) to comoving (Mpc, Mpc, Mpc)
H0 = 100.0 # Hubble constant at z = 0, [km/sec/Mpc]
Om0 = 0.30964 # Omega matter: density of non-relativistic matter in units of the critical density at z = 0
cos = FlatLambdaCDM(H0, Om0)


# frequency bin (6 bins)
freq_bins = np.loadtxt('bins_frequency.txt').astype(int)[4:5]
nfb = freq_bins.shape[0] # number of frequency bins
Rs = [ R ]
freqs = [ freq ]
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

    # nu0 = freq[len(freq)//2] # central frequency, score 102.89465978280825
    # nu0 = freq[0] # start frequency, score: 102.73098598499035
    nu0 = freq[-1] # end frequency, score: 103.7738630225836
    nu_HI = 1420.406 # MHz
    z = 1.0e6 * nu_HI / nu0 - 1
    print(z)
    rz = cos.comoving_distance(1.0e6 * nu_HI / freq[0] - 1) - cos.comoving_distance(1.0e6 * nu_HI / freq[-1] - 1) # Comoving line-of-sight distance in Mpc
    rx = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    ry = cos.comoving_transverse_distance(z) * np.radians(pix_size) * nra # Comoving transverse distance in Mpc at central freq
    print(rx, ry, rz)


    R = Rs[fbi].transpose(1, 2, 0)
    window = None
    # window = 'blackmanharris'
    # window = 'tukey'
    ps, err, kper_mid, kpar_mid, n_modes = power_spectrum_2d(R, kbins=[kper, kpar], binning=None, box_dims=[rx.value, ry.value, rz.value], return_modes=True, window=window)

    # save ps to file
    fl = f'TianlaiTest_{freq_bins[fbi][0]}MHz_{freq_bins[fbi][1]}MHz.data'
    np.savetxt(fl, ps.T, fmt='%.12f') # Note: transpose to make each row is of constant k_parallel

    # save 1sigma error to file
    fl = f'TianlaiTest_{freq_bins[fbi][0]}MHz_{freq_bins[fbi][1]}MHz_errors.data'
    np.savetxt(fl, err, fmt='%.12f')