#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:35:44 2021.

Functions to extract fitted parameter functions to simulated MRFs.
@author: kornak
"""
import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression


def invQuadMdl(avec, x, y):
    """Generate autocovariance residuals for the Gaussian MRF vs. BIFS.

    Parameters
    ----------
    avec : float numpy.ndarray
        Three parameter vector for model function residuals.
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    y : float numpy.ndarray
        Observed values.
    mdl : string
        Choice of model

    Returns
    -------
    mdlval : float numpy.ndarray
        Autocovariance residuals for the Gaussian MRF.

    """
    mdl = "invcubic"
    if mdl == "invlin":
        mdlval = (1 / (avec[0] + x * avec[1])) - y
    if mdl == "invlinsqrt":
        mdlval = (1 / (avec[0] + x * avec[1] + np.sqrt(x) * avec[2])) - y
    if mdl == "invcubic":
        mdlval = (1 / (avec[0] + x * avec[1] + x**2 * avec[2] + x**3
                       * avec[3])) - y
    if mdl == "invquadraticsqrt":
        mdlval = (1 / (avec[0] + x * avec[1] + np.sqrt(x) * avec[2]
                       + x**2 * avec[3])) - y
    if mdl == "invcubicsqrt":
        mdlval = (1 / (avec[0] + x * avec[1] + np.sqrt(x) * avec[2]
                       + x**2 * avec[3] + x**3 * avec[4])) - y
    if mdl == "invquartic":
        mdlval = (1 / (avec[0] + x * avec[1] + x**2 * avec[2] + x**3 * avec[3]
                       + x**4 * avec[4])) - y
    if mdl == "invquarticsqrt":
        mdlval = (1 / (avec[0] + (x-avec[5]) * avec[1] + np.sqrt(x-avec[5]) *
                       avec[2] + (x-avec[5])**2 * avec[3] + (x-avec[5])**3 *
                       avec[4])) - y
    if mdl == "centeredinvquarticsqrt":
        mdlval = (1 / (avec[0] + (x-avec[6]) * avec[1] + np.sqrt(x-avec[6]) *
                       avec[2] + (x-avec[6])**2 * avec[3] + (x-avec[6])**3 *
                       avec[4] + (x-avec[6])**4 * avec[5])) - y
    return mdlval


def expquadmdl(avec, x, y):
    """Generate autocovariance residuals for the Gaussian MRF vs. BIFS.

    Parameters
    ----------
    avec : float numpy.ndarray
        Five parameter vector for model function residuals.
    x : float numpy.ndarray
        Distances from center of Fourier space for each index.
    y : float numpy.ndarray
        Observed values.

    Returns
    -------
    mdlval : float numpy.ndarray
        Autocovariance residuals for the Gaussian MRF.

    """
    mdlval = avec[0] + avec[1] \
        * np.exp(-avec[2] * x - avec[3] * (x - avec[4])**2) - y
    return mdlval


def l1mdl(bvec, invx, y):
    """Generate autocovariance residuals for the L1prior MRF vs. BIFS.

    Parameters
    ----------
    bvec : float numpy.ndarray
        Two parameter vector for model function residuals.
    invx : float numpy.ndarray
        Inverse of distances from center of Fourier space for each index.
    y : float numpy.ndarray
        Observed values.

    Returns
    -------
    l1mdlval : float numpy.ndarray
        Fitted k-space mean residuals for the L1 MRF.b

    """
    l1mdlval = bvec[0] + bvec[1] * invx - y
    return l1mdlval


def invQmodelfit(kdst, gmrfKmeans, qstartPars, adim=128):
    """GMRF: Fitting fn to the mean and SD of the modulus against distance.

    Parameters
    ----------
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes of GMRF sims over Fourier space points.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfModelStartPars : float tuple, optional
        Set of start values for nonlinear optimization. The default is
        (1.0, 1.0).

    Returns
    -------
    gmrfExpParEst : float numpy.ndarray
        Estimated parameters for mean of modulus function.
    gmrfExpPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.

    """
    """ GMRF: Fitting function to the mean of the modulus against distance. """
    qExpParOpt = opt.least_squares(invQuadMdl, qstartPars,
                                   args=(kdst[1:], gmrfKmeans[1:]))
    qExpParEst = qExpParOpt['x']
    qExpPred = invQuadMdl(qExpParEst, kdst, 0)
    return qExpParEst, qExpPred


def gmrfmodelfit(kdst, Kmns, Ksds, adim=128,
                 gmrfModelStartPars=(40.0, 80.0, 0.00001, 0.00001, 0.0000001)):
    """GMRF: Fitting fn to the mean and SD of the modulus against distance.

    Parameters
    ----------
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    Kmns : float numpy.ndarray
        Means of magnitudes of GMRF sims over Fourier space points.
    Ksds : float numpy.ndarray
        SDs of magnitudes of GMRF sims over Fourier space points.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfModelStartPars : float tuple, optional
        Set of start values for nonlinear optimization. The default is
        (40.0, 80.0, 0.00001, 0.00001, 0.0000001).

    Returns
    -------
    gmrfParEst : float numpy.ndarray
        Estimated parameters for mean of modulus function.
    gmrfSDparEst : float
        Estimated parameter for SD (y) regressed on the mean (X).
    gmrfFittedSD : float numpy.ndarray
        Matrix of fitted SD estimates.
    gmrfPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.
    gmrfPredSDvals : float numpy.ndarray
        Predicted SD values based on Means of Magnitudes of FS points.

    """
    """ GMRF: Fitting function to the mean of the modulus against distance. """
    parOpt = opt.least_squares(expquadmdl, gmrfModelStartPars, args=(kdst[1:],
                                                                     Kmns[1:]))
    parEst = parOpt['x']
    pred = expquadmdl(parEst, kdst, 0)
    totalPixels = adim * adim
    Kmns.shape = (totalPixels, 1)
    Ksds.shape = (totalPixels, )
    """ GMRF: Fitting the SD of modulus function relative to the mean. """
    SDmodel = LinearRegression(fit_intercept=False)
    SDparfit = SDmodel.fit(Kmns, Ksds)
    SDparEst = SDparfit.coef_[0]
    predSD = SDmodel.predict(Kmns)
    Kmns.shape = (totalPixels, )
    fittedSD = pred * SDparEst
    return parEst, SDparEst, fittedSD, pred, predSD


def l1mrfmodelfit(kdst, invkdst, l1mrfKmeans, l1mrfKsds, adim=128):
    """L1 MRF: Fitting fn to the mean and SD of the modulus against distance.

    Parameters
    ----------
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : loat numpy.ndarray
        Matrix of inverse distances from origin in Fourier space (i.e. 1/kdst).
    l1mrfKmeans : float numpy.ndarray
        Means of magnitudes of L1 MRF sims over Fourier space points.
    l1mrfKsds : float numpy.ndarray
        SDs of magnitudes of L1 MRF sims over Fourier space points.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.

    Returns
    -------
    l1mrfParEst : float numpy.ndarray
        Estimated parameters for mean of modulus function.
    l1mrfSDparEst : float
        Estimated parameter for SD (y) regressed on the mean (X).
    l1mrfFittedSD : float numpy.ndarray
        Matrix of fitted SD estimates.
    l1mrfPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.
    l1mrfPredSDvals : float numpy.ndarray
        Predicted SD values based on Means of Magnitudes of FS points.

    """
    """ L1MRF: Fitting funcn to the mean of the modulus against distance. """
    # L1 mdl
    l1mrfModel = LinearRegression()
    totalPixels = adim * adim
    invkdst.shape = (totalPixels, 1)
    l1mrfKmeans.shape = (totalPixels, )
    l1mrfFit = l1mrfModel.fit(invkdst[1:, ], l1mrfKmeans[1:])
    l1mrfParEst = np.array([l1mrfFit.intercept_, l1mrfFit.coef_[0]])
    invkdst.shape = (totalPixels, )
    l1mrfPred = l1mdl(l1mrfParEst, invkdst, 0)
    l1mrfKmeans.shape = (totalPixels, 1)
    """ L1MRF: Fitting the SD of modulus function relative to the mean. """
    l1mrfSDmodel = LinearRegression(fit_intercept=False)
    l1mrfSDparfit = l1mrfSDmodel.fit(l1mrfKmeans, l1mrfKsds)
    l1mrfSDparEst = l1mrfSDparfit.coef_[0]
    l1mrfPredSDvals = l1mrfSDmodel.predict(l1mrfKmeans)
    l1mrfKmeans.shape = (totalPixels, )
    l1mrfFittedSD = l1mrfPred * l1mrfSDparEst
    return l1mrfParEst, l1mrfSDparEst, l1mrfFittedSD, l1mrfPred, \
        l1mrfPredSDvals


def simMRFbifs(gmrfPred, gmrfFittedSD=None, adim=128, nsamps=1000, dist="exp",
               square=False):
    """
    Generate GMRF simulations from fitted Gaussian or Exp conjugate prior.

    Old version uses full specification rather than relying on Hermitian
    symmetry. Note does not divide by adim^2 -- but does ortho normalization

    Parameters
    ----------
    gmrfPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.
    gmrfFittedSD : float numpy.ndarray
        Matrix of fitted SD estimates (can be none for Exponential)
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    nsamps : int, optional
        Number of samples of GMRF used. The default is 1000.

    Returns
    -------
    gmrfFSsim : float numpy.ndarray
        Output simulations of GMRF based on GMRF simulations from fitted
        Gaussian conjugate prior -- note that this is old version that
        evaluates full matrix in Fourier space building full Hermitian
        symmetric matrix.

    """
    hadim = int(adim/2)
    udim = hadim + 1
    np.random.seed(2)
    if dist == "exp":
        gmrfSimMapsMagtd = np.random.exponential(scale=1.0, size=(nsamps, adim,
                                                                  adim))
    if dist == "gauss":
        gmrfSimMapsMagtd = np.random.normal(loc=0.0, scale=1.0, size=(
            nsamps, adim, adim))
    np.random.seed(3)
    gmrfSimMapsPhase = np.random.uniform(low=(-np.pi), high=np.pi,
                                         size=(nsamps, adim, adim))
    gmrfSimMapsPhase[:, 0, 0] = 0.0
    if (adim % 2) == 0:
        gmrfSimMapsPhase[:, hadim, hadim] = 0.0
        gmrfSimMapsPhase[:, hadim, 0] = 0.0
        gmrfSimMapsPhase[:, 0, hadim] = 0.0
    gmrfFSsim = np.zeros((nsamps, adim, adim), dtype=complex)
    gmrfPred.shape = (adim, adim)
    if dist == "gauss":
        gmrfFittedSD.shape = (adim, adim)
    for i in range(nsamps):
        if dist == "exp":
            gmrfSimMapsMagtd[i, :, :] = \
                gmrfSimMapsMagtd[i, :, :] * gmrfPred
        if dist == "gauss":
            gmrfSimMapsMagtd[i, :, :] = \
                gmrfSimMapsMagtd[i, :, :] * gmrfFittedSD + gmrfPred
    gmrfSimMapsMagtd[:, 0, 0] = 0.0
    if square is True:
        gmrfSimMapsMagtd = np.sqrt(gmrfSimMapsMagtd)
    for i in range(nsamps):
        gmrfFSsim[i, :, :] = gmrfSimMapsMagtd[i, :, :] * np.exp(
            - 1j * gmrfSimMapsPhase[i, :, :])
    for m in range(adim):
        for n in range(udim):
            gmrfFSsim[:, m, n] = gmrfFSsim[:, -m, -n].conj()
    for i in range(nsamps):
        gmrfFSsim[i, :, :] = np.fft.ifft2(gmrfFSsim[i, :, :], norm="ortho")

    gmrfFSsim = np.real(gmrfFSsim)  # / (adim * adim)
    return gmrfFSsim


def toEstACF(simArray, adim=128, nsamps=1000):
    """
    Calculate ACF values at each kdst location in Fourier space.

    Parameters
    ----------
    simArray : float numpy.ndarray
        Array of random field simulations to estimate spatial ACF for.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    nsamps : int, optional
        Number of samples of GMRF used. The default is 1000.

    Returns
    -------
    simSpatialACF : float numpy.ndarray
        The estimated ACF over space as defined by kdst.

    """
    totalPixels = adim * adim
    hadim = int(adim/2)
    udim = hadim + 1
    simFFT = np.zeros(simArray.shape, dtype=complex)
    simFFTsq = np.zeros(simArray.shape)
    for i in range(nsamps):
        simFFT[i, :, :] = np.fft.fft2(simArray[i, :, :], norm="ortho")

    for i in range(nsamps):
        simFFTsq[i, :, :] = np.real(simFFT[i, :, :] * np.conj(simFFT[i, :, :]))

    simSpatialSq = np.mean(simFFTsq, axis=0)
    simSpatialFFTsymmetric = simSpatialSq
    for m in range(adim):
        for n in range(udim):
            simSpatialFFTsymmetric[m, n] = (simSpatialSq[m, n]
                                            + simSpatialSq[-m, -n]) / 2
    simSpatialACF = np.real(np.fft.fft2(simSpatialFFTsymmetric, norm="ortho"))
    simSpatialSq.shape = (totalPixels, )
    simSpatialACF = simSpatialACF.reshape((totalPixels, ))
    return simSpatialSq, simSpatialACF
