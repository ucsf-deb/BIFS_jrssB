#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  4 12:55:50 2021.

Read in the map and simulate the MRI brain
@author: kornak
"""
import numpy as np
import scipy.io
import bifsfns as bifs
import matplotlib.pyplot as plt

gfl = ("/Users/kornak/Documents/Projects/KspacePrior/"
       "MNIData/MNIData2D/gmask.dat")
wfl = ("/Users/kornak/Documents/Projects/KspacePrior/"
       "MNIData/MNIData2D/wmask.dat")


def simbrainMRI(adim=128, imsc=10.0, gi=2.0, wi=1.0, sdi=0.25, gfile=gfl,
                wfile=wfl, seedval=0):
    """Create simulation of the brain plus noise etc.

    Parameters
    ----------
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    imsc : float, optional
        Image scale constant for brain simulation. The default is 25.0.
    gi : float, optional
        Relative intensity for gray matter. The default is 2.0.
    wi : float, optional
        Relative intensity for white matter. The default is 1.0.
    sdi : float, optional
        Relative intensity for noise. The default is 0.3.
    gfile : string, optional
        Filename for the gray matter mask. The default is gfl.
    wfile : string, optional
        Filename for the white matter mask. The default is wfl.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    noiseSD : float
        Standard deviation of noise in image space.
    knoiseSD : float
        Estimated standard deviation of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSDEst : float
        Rayleigh-based standard deviation of modulus of noise in Fourier space.

    """
    with open(gfile, 'rb') as file:
        dat = np.fromfile(file, dtype=np.int32)
        gmsk = np.reshape(dat, [adim, adim])

    with open(wfile, 'rb') as file:
        dat = np.fromfile(file, dtype=np.int32)
        wmsk = np.reshape(dat, [adim, adim])

    totalPixels = adim * adim
    gmsk.astype(np.float64)
    wmsk.astype(np.float64)
    gmsk = np.transpose(gmsk[:, ::-1])
    wmsk = np.transpose(wmsk[:, ::-1])
    grayIntensity = gi * imsc
    whiteIntensity = wi * imsc
    MRmap = grayIntensity * gmsk + whiteIntensity * wmsk
    cleanImage = np.copy(MRmap)
    cleanImage.astype(np.float64)
    shapeImage = np.asarray(cleanImage.shape)
    np.random.seed(seedval)
    noiseSD = sdi * imsc
    noise = np.reshape(np.random.normal(0.0, noiseSD,
                                        cleanImage.size), shapeImage)
    imgPlusNoise = cleanImage + noise
    fftimgF = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimgF = np.abs(fftimgF)
    argfimgF = np.angle(fftimgF)  # Extract corresponding phase image
    kdst = bifs.kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSDest = np.std(np.abs(fftNoise), ddof=1)
    knoiseSD = np.sqrt(noiseSD**2 * (1 - np.pi/4))
    kdst.shape = (totalPixels, )
    return noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, \
        magfimgF, argfimgF, knoiseSDest, gmsk, wmsk


def simFreq(adim=128, mag1=2, mag2=1, mag3=3, freq1=0.5, freq2=0.8, freq3=0.1,
            phase1=0.5, phase2=1.0, phase3=0.1, sdi=0.25, seedval=0):
    """Create simulation of the brain plus noise etc.

    Parameters
    ----------
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    noiseSD : float
        Standard deviation of noise in image space.
    knoiseSD : float
        Estimated standard deviation of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSDEst : float
        Rayleigh-based standard deviation of modulus of noise in Fourier space.

    """

    totalPixels = adim * adim
    cleanImage = np.zeros([adim, adim])
    for i in range(adim):
        for j in range(adim):
            cleanImage[i, j] = mag1 * np.sin(i*freq1*2*np.pi - phase1) \
                + mag2 * np.sin(j*freq2*2*np.pi - phase2) \
                + mag3 * np.sin((i+j)*freq3*2*np.pi - phase3)
    cleanImage.astype(np.float64)
    shapeImage = np.asarray(cleanImage.shape)
    np.random.seed(seedval)
    noiseSD = sdi
    noise = np.reshape(np.random.normal(0.0, noiseSD,
                                        cleanImage.size), shapeImage)
    imgPlusNoise = cleanImage + noise
    fftimgF = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimgF = np.abs(fftimgF)
    argfimgF = np.angle(fftimgF)  # Extract corresponding phase image
    kdst = bifs.kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSDest = np.std(np.abs(fftNoise), ddof=1)
    knoiseSD = np.sqrt(noiseSD**2 * (1 - np.pi/4))
    kdst.shape = (totalPixels, )
    return noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, \
        magfimgF, argfimgF, knoiseSDest


def manipGmrfSim(cleanImage, gmrfl, adim=128, nsamps=1000):
    """Read Gaussian MRF simulations and performing manipulations thereof.

    A) read GMRF sims; B) take FFT of each sim; C) convert to magnitude and
    phase maps; D) calculate mean and sd of magnitude maps across sims.

    Parameters
    ----------
    cleanImage : float numpy.ndarray
        Noise free version of image of interest.
    gmrfFile : string
        Filename with the set of GMRF simulations.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    nsamps : int, optional
        Number of samples of GMRF used. The default is 1000.

    Returns
    -------
    gmrfmat : float numpy.ndarray
        adim x adim simulations times nsamps = adim x adim x nsamps array.
    gmrfsd : float
        Estimated marginal SD of zero mean GMRF.
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes over Fourier space points.
    gmrfKsds : float numpy.ndarray
        SDs of magnitudes over Fourier space points.

    """
    with open(gmrfl, 'rb') as file:
        gmrfdat = np.fromfile(file, dtype=np.float64)

    gmrfmat = np.reshape(gmrfdat, [nsamps, adim, adim])
    for i in range(nsamps):
        gmrfmat[i, :, :] = gmrfmat[i, :, :] - np.mean(gmrfmat[i, :, :])

    gmrfsd = np.std(gmrfmat, ddof=1)
    # Generating FFT based simulations (marginally indep priors in k-space)
    gmrfFFT = np.zeros(gmrfmat.shape, dtype=complex)
    for i in range(nsamps):
        gmrfFFT[i, :, :] = np.fft.fft2(gmrfmat[i, :, :], norm="ortho")

    gmrfModFFT = np.abs(gmrfFFT)
    gmrfModFFTsq = np.real(gmrfFFT * np.conj(gmrfFFT))
    gmrfKmeans = np.mean(gmrfModFFT, axis=0)
    gmrfMeanModFFTsq = np.mean(gmrfModFFTsq, axis=0)
    gmrfKsds = np.std(gmrfModFFT, axis=0, ddof=1)
    gmrfSDevModFFTsq = np.std(gmrfModFFTsq, axis=0)
    totalPixels = adim * adim
    gmrfKmeans.shape = (totalPixels, )
    gmrfMeanModFFTsq.shape = (totalPixels, )
    gmrfKsds.shape = (totalPixels, )
    gmrfSDevModFFTsq.shape = (totalPixels, )
    return gmrfmat, gmrfsd, gmrfKmeans, gmrfKsds, gmrfFFT, gmrfModFFT, \
        gmrfModFFTsq, gmrfMeanModFFTsq, gmrfSDevModFFTsq


def manipL1mrfSim(cleanImage, l1mrfl, adim=128, l1nsamps=1000):
    """Read L1 MRF simulations and performing manipulations thereof.

    A) read L1 MRF sims; B) take FFT of each sim; C) convert to magnitude and
    phase maps; D) calculate mean and sd of magnitude maps across sims.

    Parameters
    ----------
    cleanImage : float numpy.ndarray
        Noise free version of image of interest.
    l1mrfl : string
        Filename with the set of L1 MRF simulations.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    l1nsamps : int, optional
        Number of samples of L1 MRF used. The default is 1000.

    Returns
    -------
    l1mrfmat : float numpy.ndarray
        adim x adim simulations times nsamps = adim x adim x nsamps array.
    l1mrfsd : float
        Estimated marginal SD of zero mean L1 MRF.
    l1mrfKmeans : float numpy.ndarray
        Means of magnitudes over Fourier space points.
    l1mrfKsds : float numpy.ndarray
        SDs of magnitudes over Fourier space points.

    """
    test = scipy.io.loadmat(l1mrfl)
    l1mrf = test['RFsamps']
    l1mrfmat = np.zeros((l1nsamps, adim, adim))

    for i in range(l1nsamps):
        l1mrfmat[i, :, :] = l1mrf[:, :, i] - np.mean(l1mrf[:, :, i])

    l1mrfsd = np.std(l1mrfmat, ddof=1)

    # Generating FFT-based simulations (marginally indep priors in FS)
    l1mrfFFT = np.zeros(l1mrfmat.shape, dtype=complex)
    for i in range(l1nsamps):
        l1mrfFFT[i, :, :] = np.fft.fft2(l1mrfmat[i, :, :], norm="ortho")

    l1ModFFT = np.abs(l1mrfFFT)
    l1ModFFTsq = np.real(l1mrfFFT * np.conj(l1mrfFFT))
    l1mrfKmeans = np.mean(l1ModFFT, axis=0)
    l1MeanModFFTsq = np.mean(l1ModFFTsq, axis=0)
    l1mrfKsds = np.std(l1ModFFT, axis=0, ddof=1)
    l1SDevModFFTsq = np.std(l1ModFFTsq, axis=0)
    totalPixels = adim * adim
    l1mrfKmeans.shape = (totalPixels, )
    l1MeanModFFTsq.shape = (totalPixels, )
    l1mrfKsds.shape = (totalPixels, )
    l1SDevModFFTsq.shape = (totalPixels, )
    return l1mrfmat, l1mrfsd, l1mrfKmeans, l1mrfKsds, \
        l1ModFFTsq, l1MeanModFFTsq, l1SDevModFFTsq


def plotGmrfFits(kdst, gmrfKmeans, gmrfPred, gmrfKsds, gmrfPredSDvals,
                 l1mrfKmeans, l1mrfPred, l1mrfKsds, l1mrfPredSDvals, s1=2,
                 s2=2, figsize=(10, 10)):
    """Generate plots of autocovariance/spectral power fits.

    Parameters
    ----------
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space..
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes over Fourier space points.
    gmrfPred : float numpy.ndarray
        Map of parameter estimates for gmrfKmeans regressed on kdst.
    gmrfKsds : float numyy.ndarray
        SDs of magnitudes over Fourier space points.
    gmrfPredSDvals : float numpy.ndarray
        Predicted parameter estimates for fitting SD relative to mean.
    l1mrfKmeans : TYPE
        DESCRIPTION.
    l1mrfPred : TYPE
        DESCRIPTION.
    l1mrfKsds : TYPE
        DESCRIPTION.
    l1mrfPredSDvals : TYPE
        DESCRIPTION.
    s1 : TYPE, optional
        DESCRIPTION. The default is 2.
    s2 : TYPE, optional
        DESCRIPTION. The default is 2.
    figsize : TYPE, optional
        DESCRIPTION. The default is (10, 10).

    Returns
    -------
    None.

    """
    gmrfPred.shape = (np.prod(gmrfPred.shape), )
    gmrfPredSDvals.shape = (np.prod(gmrfPredSDvals.shape), )
    fig, axs = plt.subplots(s1, s2, figsize=figsize)
    axs[0, 0].scatter(kdst, gmrfKmeans, s=0.3, c='b', label='GMRF kmeans')
    axs[0, 0].plot(kdst, gmrfPred, c='r', label='python fit')
    # axs[0, 0].plot(kdst, gmrfPred2, c='g', label='matlab fit')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Distance from k-space origin')
    axs[0, 0].set_ylabel('Mean of magnitude')
    axs[0, 0].set_title('GMRF Modulus fit')
    axs[0, 1].scatter(gmrfKmeans, gmrfKsds, s=0.3, c='b', label='observed SD')
    axs[0, 1].plot(gmrfKmeans, gmrfPredSDvals, c='r', label='fitted SD')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Mean of GMRF k-space magnitude')
    axs[0, 1].set_ylabel('Standard deviation')
    axs[0, 1].set_title('GMRF fit SD')
    axs[1, 0].scatter(kdst, l1mrfKmeans, s=0.3, c='b', label='L1 kmeans')
    axs[1, 0].plot(kdst[1:], l1mrfPred, c='g', label='fitted')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Distance from k-space origin')
    axs[1, 0].set_ylabel('Mean of magnitude')
    axs[1, 0].set_title('L1 MRF Modulus fit')
    axs[1, 1].scatter(l1mrfKmeans, l1mrfKsds, s=0.3, c='b',
                      label='observed SD')
    axs[1, 1].plot(l1mrfKmeans, l1mrfPredSDvals, c='r', label='fitted SD')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Mean of L1 MRF k-space magnitude')
    axs[1, 1].set_ylabel('Standard deviation')
    axs[1, 1].set_title('L1 MRF fit SD')
    plt.show()
