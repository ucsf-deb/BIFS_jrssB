#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 25 13:43:39 2021.

BIFS functions for processing steps.
@author: kornak
"""
import scipy.special as special
import scipy.ndimage.filters as fltr
import numpy as np


def pltout(Image, imgset, outdir):
    for i in range(np.shape(imgset)[0]):
        temp1 = c2g(imgset[i])
        temp2 = Image.fromarray(temp1)
        temp2.save(outdir + "img" + str(i) + ".pdf")


def c2g(img):
    """Generate rescaled image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be rescaled as 0 to 256.

    Returns
    -------
    x : numpy.ndarray
        Rescaled image between 0 and 256.

    """
    x = (((img - img.min()) / (img.max() - img.min()))
         * 255.9).astype(np.uint8)
    return x


def c2gAdjust(img, minv, maxv):
    """Generate rescaled image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be rescaled as 0 to 256.

    Returns
    -------
    x : numpy.ndarray
        Rescaled image between 0 and 256.

    """
    x = (((img - minv) / (maxv - minv))
         * 255.9).astype(np.uint8)
    return x


def genFSdata(imgarr, noiseSD=1.0, seedval=0):
    """Generate FS data adding noise in image space.

    Parameters
    ----------
    imgarr : float numpy.ndarray
        Array containing the clean image intensities.
    noiseSD : float, optional
        Standard deviation of noise to be added in image space
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimg : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimg : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.

    """
    imgarr.astype(np.float64)
    cleanImage = np.copy(imgarr)
    np.random.seed(seedval)
    noise = np.reshape(np.random.normal(0.0, noiseSD, cleanImage.size),
                       cleanImage.shape)
    imgPlusNoise = cleanImage + noise
    fftimg = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimg = np.abs(fftimg)
    argfimg = np.angle(fftimg)  # Extract corresponding phase image
    kdst = kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSD = np.std(np.abs(fftNoise), ddof=1)
    logknoiseSD = np.std(np.log(np.abs(fftNoise)), ddof=1)
    knoiseMean = np.mean(np.abs(fftNoise))
    logknoiseMean = np.mean(np.log(np.abs(fftNoise)))
    return knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg, \
        argfimg, logknoiseSD, knoiseMean, logknoiseMean


def genFSdataTdist(imgarr, noiseSD=1.0, tdf=3, seedval=0):
    """Generate FS data adding noise in image space.

    Parameters
    ----------
    imgarr : float numpy.ndarray
        Array containing the clean image intensities.
    noiseSD : float, optional
        Standard deviation of noise to be added in image space
    tdf : int, optional
        Degrees of freedom for t-distributed noise.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimg : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimg : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.

    """
    imgarr.astype(np.float64)
    cleanImage = np.copy(imgarr)
    np.random.seed(seedval)
    noise = np.reshape(np.sqrt((tdf - 2)/tdf) * noiseSD * np.random.standard_t(
        df=tdf, size=cleanImage.size), cleanImage.shape)
    imgPlusNoise = cleanImage + noise
    fftimg = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimg = np.abs(fftimg)
    argfimg = np.angle(fftimg)  # Extract corresponding phase image
    kdst = kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSD = np.std(np.abs(fftNoise), ddof=1)
    logknoiseSD = np.std(np.log(np.abs(fftNoise)), ddof=1)
    knoiseMean = np.mean(np.abs(fftNoise))
    logknoiseMean = np.mean(np.log(np.abs(fftNoise)))
    return knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg, \
        argfimg, logknoiseSD, knoiseMean, logknoiseMean


def genFSdataTdistApprox(imgarr, noiseSD=1.0, tdf=3, seedval=0):
    """Generate FS data adding noise in image space.

    Parameters
    ----------
    imgarr : float numpy.ndarray
        Array containing the clean image intensities.
    noiseSD : float, optional
        Standard deviation of noise to be added in image space
    tdf : int, optional
        Degrees of freedom for t-distributed noise.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimg : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimg : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.

    """
    imgarr.astype(np.float64)
    cleanImage = np.copy(imgarr)
    np.random.seed(seedval)
    noise = np.reshape(np.random.standard_t(
        df=tdf, size=cleanImage.size), cleanImage.shape)
    noiseSTD = np.std(noise, ddof=1)
    noise = (noiseSD/noiseSTD) * noise
    imgPlusNoise = cleanImage + noise
    fftimg = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimg = np.abs(fftimg)
    argfimg = np.angle(fftimg)  # Extract corresponding phase image
    kdst = kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSD = np.std(np.abs(fftNoise), ddof=1)
    logknoiseSD = np.std(np.log(np.abs(fftNoise)), ddof=1)
    knoiseMean = np.mean(np.abs(fftNoise))
    logknoiseMean = np.mean(np.log(np.abs(fftNoise)))
    return knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg, \
        argfimg, logknoiseSD, knoiseMean, logknoiseMean


def bessd(x, limval=700.0):
    """
    Calculate ratio of modified Bessel functions.

    Specifically ratio of modified Bessel function of first kind order 1
    divided by modified Bessel function of first kind order 0.

    Parameters
    ----------
    x : float, numpy.ndarray
        Value to calculate ratio of Bessel functions for.
    limval : float
        Value above which to assign ratio of 1 to avoid numerical problems.

    Returns
    -------
    y : float, numpy.ndarray
        Value for ratio of modified Bessel functions.

    """
    y = np.where(x > limval, 1, special.iv(1, x)/special.iv(0, x))
    return y


def kdist(n):
    """Create 1D centered k-space indexes 0:(n-1).

    Parameters
    ----------
    n : int scalar
        size of array to be generated

    Returns
    -------
    kval : float numpy.ndarray
           array length n that gives 1D distance from wrapped origin
           at each index
    """
    kval = np.zeros(n, dtype=float)
    if (n % 2) == 0:
        kval[0:(n//2)] = np.arange(n/2)
        kval[(n//2):n] = np.arange(n/2, 0, -1)
    else:
        kval[0:(1 + n//2)] = np.arange(np.ceil(n/2))
        kval[(1 + n//2):n] = np.arange(np.floor(n/2), 0, -1)
    return kval


def kdist2D(n1, n2):
    """Generate matrix of distances from center of Fourier space \
    but shifted so origin is at index (0,0) of the matrix.

    Parameters
    ----------
    n1 : int scalar
         dimension of image in x-direction
    n2 : int scalar
         dimension of image in y-direction

    Returns
    -------
    Xv : float numpy.ndarray
         n1 x n2 array with distances from center of Fourier space, but
         shifted so that origin is at index (0,0) of the array.
         Leads to in-place version of np.sqrt(Xv ** 2 + Yv ** 2)
         after np.meshgrid.
    """
    xvec = kdist(n1)
    yvec = kdist(n2)
    Xv, Yv = np.meshgrid(xvec, yvec, indexing='ij')
    Xv **= 2
    Yv **= 2
    Xv += Yv
    Xv = np.sqrt(Xv)
    return Xv


def indxs(sx1, sx2, rad=3):
    """Generate indices for a center circle of k-space with radius 'rad'\
    a 2D array of dimension sx1 by sx2.

    Parameters
    ----------
    sx1 : int
        First dimension of 2D array.
    sx2 : int
        Second dimension of 2D array.
    rad : float, optional
        Radius of center circle to generate. The default is 3.

    Returns
    -------
    coords : int numpy.ndarray
        Array of coordinates generetaed.

    """
    d1 = sx1 / 2 - 0.5
    d2 = sx2 / 2 - 0.5
    coords = np.empty(shape=(0, 2), dtype=np.int32)
    for i in range(sx1):
        for j in range(sx2):
            it = i
            jt = j
            if i > d1:
                it = it - sx1
            if j > d2:
                jt = jt - sx2
            testdist = np.sqrt(it ** 2 + jt ** 2)
            if testdist <= rad + 1e-5:
                coords = np.concatenate((coords, np.array([[i, j]])), axis=0)
    return coords


def flatten(x, cntrvals):
    """Flatten Fourier space in the set indices cntrvals to lowest value on \
    the perimeter.

    Parameters
    ----------
    x : float numpy.ndarray
        Array representing unflattened parameter function.
    cntrvals : int numpy.ndarray
        Indices of locations of Fourier space to be flattened.

    Returns
    -------
    y : float numpy.ndarray
        Center flattened version of parameter function.

    """
    y = x.copy()
    minval = np.max(y)
    for i in range(cntrvals.shape[0]):
        minval = np.min([minval, y[cntrvals[i, 0], cntrvals[i, 1]]])
    for i in range(cntrvals.shape[0]):
        y[cntrvals[i, 0], cntrvals[i, 1]] = minval
    return y


def pwr_ratio(mat1, mat2):
    """
    Return power in matrix but subtracting the mean (i.e. without (0,0) freq.

    Parameters
    ----------
    mat1 : float numpy.ndarray
        Matrix to determine intrinsic power of.
    mat2 : float numpy.ndarray
        Matrix to determine intrinsic power of.

    Returns
    -------
    x : float
        Ratio of power first to second matrix minus FS at zero frequency.

    """
    x1 = np.sum(np.reshape(mat1**2, -1)[1:]) / np.prod(mat1.shape)
    x2 = np.sum(np.reshape(mat2**2, -1)[1:]) / np.prod(mat2.shape)
    x = np.sqrt(x1/x2)
    return x


"""
Univariate functions to calculate posterior estimates at a single location in
Fourier space
"""


def gauss_gauss_post(r, sig, m, s, zc=False):
    """Generate "Gaussian posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood using conjugate
    Bayes

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.
    s : float numpy.ndarray
        Prior value(s) for the sd of the FS signal magnitude.
    zc : bool, optional
        Flag to indicate whether to correct for potential zeros in the prior
        SD. The default is False.

    Returns
    -------
    rho : float numpy.ndarray
        Gaussian posterior mode BIFS estimate FS signal magnitude -- the
        conventional Gaussian prior, Gaussian likelihood conjugate Bayes
        formulation when both mean and SD are unknown.

    """
    minval = 1e-12
    sig2 = sig ** 2
    s2 = s ** 2
    rho = (m/s2 + r/sig2)/(1/s2 + 1/sig2)
    if zc:
        test = s < minval
        rho[test] = minval  # set to minval where prior sd ~ 0
    return rho


def gauss_gauss_n_post(r, sig, m, s, nval, zc=False):
    """Generate "Gaussian posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood using conjugate
    Bayes

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.
    s : float numpy.ndarray
        Prior value(s) for the sd of the FS signal magnitude.
    nval : float numpy.ndarray
        Prior value(s) for the number of equivalent samples to count the prior
    zc : bool, optional
        Flag to indicate whether to correct for potential zeros in the prior
        SD. The default is False.

    Returns
    -------
    rho : float numpy.ndarray
        Gaussian posterior mode BIFS estimate FS signal magnitude -- the
        conventional Gaussian prior, Gaussian likelihood conjugate Bayes
        formulation when both mean and SD are unknown.

    """
    minval = 1e-12
    sig2 = sig ** 2
    s2 = s ** 2
    rho = (m*nval/s2 + r/sig2)/(nval/s2 + 1/sig2)
    if zc:
        test = s < minval
        rho[test] = minval  # set to minval where prior sd ~ 0
    return rho


def lnorm_lnorm_post(r, sig, m, s, zc=False):
    """Generate "lognorm prior / Gaussian likelihood" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood using conjugate
    Bayes

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    logsig : float numpy.ndarray
        Likelihood value(s) for the sd of the signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.
    s : float numpy.ndarray
        Prior value(s) for the sd of the FS signal magnitude.
    zc : bool, optional
        Flag to indicate whether to correct for potential zeros in the prior
        SD. The default is False.

    Returns
    -------
    rho : float numpy.ndarray
        Gaussian posterior mode BIFS estimate FS signal magnitude -- the
        conventional Gaussian prior, Gaussian likelihood conjugate Bayes
        formulation when both mean and SD are unknown.

    """
    minval = 1e-12
    logr = np.log(r)
    sig2 = sig ** 2
    s2 = s ** 2
    rho = (m/s2 + r/sig2)/(1/s2 + 1/sig2)
    if zc:
        test = s < minval
        rho[test] = minval  # set to minval where prior sd ~ 0
    return rho


def exp_gauss_post(r, sig, m):
    """Generate Exp/Gauss posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood for Exponential
    prior and Gaussian likelihood.

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.

    Returns
    -------
    rho : float numpy.ndarray
        Exp/Gauss posterior mode BIFS estimate FS signal magnitude -- the
        Exponential prior with Gaussian likelihood.

    """
    sig2 = sig ** 2
    rho = r - sig2/m
    rho = np.where(rho < 0.0, 0.0, rho)
    return rho


def gauss_rice_post(r, sig, m, s, its=10):
    """Generate "Gauss/Rice posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood for Gaussian prior
    and Rician likelihood.

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.
    s : float numpy.ndarray
        Prior value(s) for the sd of the FS signal magnitude.
    its : integer, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    rho : float numpy.ndarray
        Gauss/Rice posterior mode BIFS estimate FS signal magnitude -- the
        Gaussian prior with Rician likelihood.

    """
    sig2 = sig ** 2
    s2 = s ** 2
    rho = r  # (r + m)/2
    risig2 = r / sig2
    msig2 = m * sig2
    rs2 = r * s2
    s2plssig2 = s2 + sig2
    for i in range(its):
        b = bessd(rho * risig2)
        rho = (b * rs2 + msig2) / s2plssig2
    return rho


def exp_rice_post(r, sig, m, its=10):
    """Generate "Exp/Rice posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood for Exponential
    prior and Rician likelihood.

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the mode of the FS signal magnitude.
    its : integer, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    rho : float numpy.ndarray
        Exp-Rice posterior mode BIFS estimate FS signal magnitude -- the Exp
        prior with Rician likelihood.

    """
    sig2 = sig ** 2
    rho = r
    risig2 = r / sig2
    sig2im = sig2 / m
    for i in range(its):
        b = bessd(rho * risig2)
        rho = b * r - sig2im
        rho = np.where(rho < 0.0, 0.0, rho)
    return rho


def expsq_rice_post(r, sig, m, its=100):
    """Generate "ExpSq/Rice posterior" for modulus in k-space.

    i.e. for mean of the modulus from prior and likelihood for exponential
    square prior and Rician likelihood.

    Parameters
    ----------
    r : float numpy.ndarray
        Likelihood value(s) for the mode of the FS signal magnitude.
    sig : float numpy.ndarray
        Likelihood value(s) for the sd of the FS signal magnitude.
    m : float numpy.ndarray
        Prior value(s) for the square of the mode of the FS signal magnitude.
    its : integer, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    rho : float numpy.ndarray
        ExpSq/Rice posterior mode BIFS estimate FS signal magnitude -- the
        exponential square prior with Rician likelihood.

    """
    sig2 = sig ** 2
    rho = r
    risig2 = r / sig2
    rm = r * m
    rmsq = rm ** 2
    foursig2 = 4.0 * sig2
    extra = 8.0 * (sig2 ** 2) * m + foursig2 * (m ** 2)
    denom = foursig2 + 2.0 * m
    for i in range(its):
        b = bessd(rho * risig2)
        rho = (b * rm + np.sqrt((b ** 2) * rmsq + extra)) / denom
    return rho


"""
Parameter function generation
"""


def invxy(bvec, x, exponent=2, scl=1.0, normimg=None):
    """Generate parameter function for modulus of mean as ~ 1/(const + dist^y).

    Parameter function is scl/(const + distance to power y) from origin,
    i.e. function for prior parameter function ~ b0 + b1/dist^y where the
    dist is from center of Fourier space.

    Parameters
    ----------
    bvec : float tuple
        Two parameter vector for b0.
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    exponent : float/int
        Exponent of inverse decay.
    scl : float, optional
        Factor to scale whole parameter function by. The default is 1.0.
    normimg : float numpy.ndarray
        image with respect to normalize power to

    Returns
    -------
    fval : float numpy.ndarray
        Parameter function values for corresponding x values.

    """
    fval = 1.0 / (bvec[0] + bvec[1] * x**exponent)
    if normimg is not None:
        scl = pwr_ratio(normimg, fval)
    fval = scl * fval
    return fval


def torus(x, lwr, upr, normimg=None):
    """Generate parameter function that is an elevated torus.

    Parameters
    ----------
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    lwr : int
        Distance from origin of smallest non-zero frequency.
    upr : int
        Distance from origin of largest non-zero frequency.
    normimg : float numpy.ndarray
        image with respect to normalize power to

    Returns
    -------
    fval : float numpy.ndarray
        Parameter function values for corresponding x values.

    """
    condition = np.logical_and(lwr <= x, x <= upr)
    msk = np.zeros(x.shape)
    msk[condition] = 1.0
    scl = 1.0
    if normimg is not None:
        scl = pwr_ratio(normimg, np.ones(x.shape))
    fval = msk * scl
    return fval


def torus_gauss(x, lwr, upr, sigma, normimg=None):
    """Generate parameter function that is an elevated torus with Gaussian
    smoothed edges.

    Parameters
    ----------
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    lwr : int
        Distance from origin of smallest non-zero frequency.
    upr : int
        Distance from origin of largest non-zero frequency.
    normimg : float numpy.ndarray
        image with respect to normalize power to
    sigma : float
        standard deviation for Gaussian filter

    Returns
    -------
    fval : float numpy.ndarray
        Parameter function values for corresponding x values.

    """
    condition = np.logical_and(lwr <= x, x <= upr)
    msk = np.zeros(x.shape)
    msk[condition] = 1.0
    scl = 1.0
    msk = fltr.gaussian_filter(msk, sigma=sigma)
    if normimg is not None:
        scl = pwr_ratio(normimg, np.ones(x.shape))
    fval = msk * scl
    return fval


def ixscBanded(bvec, x, y, z):
    """Generate parameter function for modulus of mean that is 1/distance \
    to power y from origin + constant i.e. function for prior parater \
    function ~ b0 + b1/dist^y where the dist is from center of Fourier \
    space and distances are truncated so function 0 after distance z.

    Parameters
    ----------
    bvec : float numpy.ndarray
        Two parameter vector for b0.
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    y : float
        Exponent of inverse decay.
    z : float
        Distance from origin for which to truncate beyond.

    Returns
    -------
    fval : float numpy.ndarray
        Truncated parameter function values for corresponding x values.

    """
    idx1 = x < z
    idx2 = bvec[0] + bvec[1]*(x**(-y))
    fval = idx1 * idx2
    return fval


def linsc(bvec, x, exponent=1):
    """Generate parameter function for modulus of mean corresponding to \
    linear decay from center of Fourier space (until crosses zero) and then \
    truncated at zero.

    Parameters
    ----------
    bvec : float numpy.ndarray
        Two parameter vector for b0.
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    exponent : float/int
        Exponent of inverse decay.

    Returns
    -------
    fval : float numpy.ndarray
        Linear decay parameter function values for corresponding x values.

    """
    fval = np.maximum(0, bvec[1] - bvec[0] * x**exponent)
    return fval


"""
Fourier space BIFS analysis parameter functions implementation.
"""


def bifs_post_mode(magfimg, argfimg, knoiseSD, meanfn, sdfn=None,
                   dist="gauss_gauss"):
    """
    Perform posterior estimation according to prior and likelihood choice.

    Parameters
    ----------
    magfimg : float numpy.ndarray
        Matrix of image magnitude values at each point in Fourier space.
    argfimg : float numpy.ndarray
        Matrix of image phase/argument values at each point in Fourier space.
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    dist : string, optional
        Prior/likelihood combination for BIFS posterior opt. The default is
        "GausssGauss".

    Returns
    -------
    imgrecon : float numpy.ndarray
        Matrix of BIFS posterior image values.

    """
    if dist == "gauss_gauss":
        kPost = gauss_gauss_post(magfimg, knoiseSD, meanfn, sdfn)
    if dist == "lnorm_lnorm":
        kPost = lnorm_lnorm_post(magfimg, knoiseSD, meanfn, sdfn)
    if dist == "exp_gauss":
        kPost = exp_gauss_post(magfimg, knoiseSD, meanfn)
    if dist == "gauss_rice":
        kPost = gauss_rice_post(magfimg, knoiseSD, meanfn, sdfn)
    if dist == "exp_rice":
        kPost = exp_rice_post(magfimg, knoiseSD, meanfn)
    if dist == "expsq_rice":
        kPost = expsq_rice_post(magfimg, knoiseSD, meanfn)
    kPost[0, 0] = magfimg[0, 0]
    imgrecon = np.real(np.fft.ifft2(kPost * np.exp(1j * argfimg),
                                    norm="ortho"))
    return imgrecon
