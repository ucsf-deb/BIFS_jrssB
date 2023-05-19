#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:55:36 2021.

Functions for MRF conjugate gradients reconstruction
@author: kornak
"""

import numpy as np


def icmEdge1(sigma2, img, kappa, delta, itrns=100):
    # icm for Gaussian prior
    xdim = img.shape[0]
    ydim = img.shape[1]
    kapsig2invdelta2 = kappa * sigma2 / (delta ** 2)
    invdelta2 = 1 / (delta ** 2)
    ximgnew = np.copy(img)
    bimg = np.zeros(img.shape)
    itrnsdble = 2 * itrns
    testval = 1
    for indx in range(itrnsdble):
        ximg = np.copy(ximgnew)
        testval = 1 - testval
        for i in range(xdim):
            for j in range(ydim):
                bimg[i, j] = (ximg[i, j] - ximg[((i - 1) % xdim), j]) / \
                    (1 + invdelta2 * (ximg[i, j] - ximg[((i - 1) % xdim), j])
                     ** 2) ** 2 + \
                             (ximg[i, j] - ximg[((i + 1) % xdim), j]) / \
                    (1 + invdelta2 * (ximg[i, j] - ximg[((i + 1) % xdim), j])
                     ** 2) ** 2 + \
                             (ximg[i, j] - ximg[i, ((j - 1) % ydim)]) / \
                    (1 + invdelta2 * (ximg[i, j] - ximg[i, ((j - 1) % ydim)])
                     ** 2) ** 2 + \
                             (ximg[i, j] - ximg[i, ((j + 1) % ydim)]) / \
                    (1 + invdelta2 * (ximg[i, j] - ximg[i, ((j + 1) % ydim)])
                     ** 2) ** 2
        for i in range(xdim):
            for j in range(ydim):
                ipj = i + j
                if ipj % 2 == testval:
                    ximgnew[i, j] = img[i, j] + kapsig2invdelta2 * bimg[i, j]
    return ximgnew


def icmGauss(sigma2, img, kappa, itrns=100):
    # icm for Gaussian prior
    xdim = img.shape[0]
    ydim = img.shape[1]
    kapsig2 = kappa * sigma2
    den = 4 * kapsig2 + 1
    ximg = np.copy(img)
    for indx in range(itrns):
        for i in range(xdim):
            for j in range(ydim):
                ximg[i, j] = (kapsig2 * (ximg[((i - 1) % xdim), j] +
                                         ximg[((i + 1) % xdim), j] +
                                         ximg[i, ((j - 1) % ydim)] +
                                         ximg[i, ((j + 1) % ydim)]) +
                              img[i, j]) / den
    return ximg


def icmMedian(sigma2, img, kappa, itrns, nbsz=5):
    # icm for absolute prior
    xdim = img.shape[0]
    ydim = img.shape[1]
    kapsig2 = kappa * sigma2
    dblkapsig2 = 2 * kappa * sigma2
    imgprop = np.zeros([xdim, ydim, nbsz])  # nbsz = # pixels in neighborhood
    imgprop[:, :, 0] = img - dblkapsig2
    imgprop[:, :, 1] = img - kapsig2
    imgprop[:, :, 2] = img
    imgprop[:, :, 3] = img + kapsig2
    imgprop[:, :, 4] = img + dblkapsig2
    ximgnew = np.copy(img)
    itrnsdble = 2 * itrns
    testval = 1
    testpost = np.zeros(nbsz)
    for indx in range(itrnsdble):
        ximg = np.copy(ximgnew)
        testval = 1 - testval
        for i in range(xdim):
            for j in range(ydim):
                ipj = i + j
                if ipj % 2 == testval:
                    testvec = np.array([ximg[((i - 1) % xdim), j],
                                        ximg[((i + 1) % xdim), j],
                                        ximg[i, ((j - 1) % ydim)],
                                        ximg[i, ((j + 1) % ydim)]])
                    for k in range(nbsz):
                        testpost[k] = (img[i, j] - imgprop[i, j, k]) ** 2 + \
                            kappa * sigma2 * np.sum(np.abs(imgprop[i, j, k] -
                                                           testvec))
                        minloc = np.argmin(testpost)
                        ximgnew[i, j] = imgprop[i, j, minloc]
    return ximgnew


def udiff(x):
    """Calculate the sum of the clique functions for use in the Gibbs \
    distribution for use in conjugate gradients.

    Parameters
    ----------
    x : float numpy.ndarray
        Values of array on which to calculate differences.

    Returns
    -------
    z : float numpy.ndarray
        Array of clique sums.

    """
    z = 4 * x - np.roll(x, 1, 0) - np.roll(x, -1, 0) - \
        np.roll(x, 1, 1) - np.roll(x, -1, 1)
    return z


def usq(x):
    """Calculate the sum of the squares of the clique functions in the Gibbs \
    distribution for use in conjugate gradients.

    Parameters
    ----------
    x : float
        Values of array on which to calculate sum of squares.

    Returns
    -------
    z : float numpy.ndarray
        Array of sum of squares.

    """
    z = (x - np.roll(x, 1, 0))**2 + (x - np.roll(x, -1, 0))**2 + \
        (x - np.roll(x, 1, 1))**2 + (x - np.roll(x, -1, 1))**2
    return z


def conjgrad(sigma2, img, itau2, itrns=1000, seedval=1):
    """Perform conjugate gradients (CG) Gaussian Markov Random Field (GMRF) \
    modulus prior and likelihood modeling with unknown mean and variance.

    Parameters
    ----------
    sigma2 : float
        standard deviation of the noise in Fourier space for magnitude \
        i.e. likelihood SD hyper-parameter start value.
    img : float numpy.ndarray
        image data to be reconstructed.
    itau2 : float
        Inverse "variance" parameter of GMRF prior, i.e. hyper-parameter for \
        prior precision.
    itrns : int, optional
        The number of conjugate gradients iterations to perform. The default \
        is 1000.
    seedval : int, optional
        Random seed value. The default is 1.

    Returns
    -------
    conjg : float numpy.ndarray
        Bayesian image analysis CG reconstructed image with GMRF prior.

    """
    xdim = img.shape[0]
    ydim = img.shape[1]
    lpxls = xdim * ydim
    # initializing
    conjG = img  # Start values for Conjugate gradients
    rmat = np.zeros((xdim, ydim))
    # Setting up parts of the prior
    vrCCx = np.zeros((xdim, ydim))
    BCCB = np.zeros((xdim, ydim))
    vr = sigma2 * itau2
    goldvec = np.zeros(lpxls)
    bvec = np.zeros(lpxls)
    itrnsp1 = itrns + 1
    for ix in range(itrnsp1):  # conjugate gradients iterations
        vrCCx = vr * udiff(conjG)  # CCx = Udiff(conjG)
        gnew = rmat - vrCCx
        gnewvec = np.concatenate(gnew)
        if ix > 0.5:
            gmma = np.dot(gnewvec, gnewvec) / np.dot(goldvec, goldvec)
        else:
            gmma = 0.0

        if ix > 0.5:
            bvec = gnewvec + gmma * bvec
        else:
            bvec = gnewvec

        bmat = np.reshape(bvec, (xdim, ydim))
        qmat = bmat
        qvec = np.concatenate(qmat)
        BCCB = usq(qmat)
        vrbccb = vr * np.sum(BCCB)
        alph = np.dot(bvec, gnewvec) / (np.dot(qvec, qvec) + vrbccb)
        conjG = conjG + alph * bmat
        rmat = rmat - alph * qmat
        goldvec = gnewvec

    return conjG
