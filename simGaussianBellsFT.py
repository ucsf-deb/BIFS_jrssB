#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:34:08 2022

Copy of creating simulations translated from matlab

@author: kornak
"""

import numpy as np
from scipy.stats import multivariate_normal
import plotfns as pltfn
import bifsfns as bifs
import postfns as pst
import mrfconjgrad as mcg

np.random.seed(2)
dim1 = 256
dim2 = dim1
dimmar = 10
dim1L = dimmar
dim2L = dimmar
dim1H = dim1 - dimmar
dim2H = dim2 - dimmar

noiseSD = 0.01
knoiseSD = np.sqrt(noiseSD**2 * (1 - np.pi/4))
poissmean = 6
simsamp = 10000
testsamp = 100
sims = simsamp + testsamp
lowrad = 4
highrad = 8
nobs = np.random.poisson(poissmean, sims)
nmax = np.max(nobs)
gausspars = np.zeros((nmax, 6))  # four columns for 2 sets of Gaussians
# plus correlation and scale (for height of distribution)
simtemp = np.zeros((dim1, dim2))
simFinal = np.zeros((dim1, dim2))
pattern = np.zeros((dim1, dim2, sims))
covmat = np.zeros(2)
meanmat = np.zeros((dim2, 2))
meanmat[:, 1] = np.arange(dim2)

for sim in range(sims):
    nobc = nobs[sim]  # number of observations for current sim
    gausspars = 0*gausspars
    gausspars[0:nobc, 0] = np.random.uniform(dim1L, dim1H, nobc)  # cntr x dim
    gausspars[0:nobc, 1] = np.random.uniform(dim2L, dim2H, nobc)  # cntr y dim
    gausspars[0:nobc, 2] = np.random.uniform(lowrad, highrad, nobc)  # sd 1st
    gausspars[0:nobc, 3] = np.random.uniform(lowrad, highrad, nobc)  # sd 2nd
    gausspars[0:nobc, 4] = np.random.uniform(0, 1, nobc)  # correlation
    gausspars[0:nobc, 5] = np.random.uniform(2, 5, nobc)  # scl to multply bell
    for no in range(nobc):
        simtemp = 0*simtemp
        covval = gausspars[no, 4] * gausspars[no, 2] * gausspars[no, 3]
        covmat = np.array([[gausspars[no, 2]**2, covval],
                           [covval, gausspars[no, 3]**2]])
        for i in range(dim1):
            meanmat[:, 0] = i
            simtemp[i, :] = multivariate_normal.pdf(
                meanmat, mean=gausspars[no, 0:2], cov=covmat)
        maxcut = np.max(simtemp) / 20
        simtemp[simtemp < maxcut] = 0.0
        simtemp = gausspars[no, 5] * simtemp
        pattern[:, :, sim] = pattern[:, :, sim] + simtemp

images = [pattern[:, :, 0], pattern[:, :, 1], pattern[:, :, 2],
          pattern[:, :, 3], pattern[:, :, 4], pattern[:, :, 5]]
plotTitles = ['sim0', 'sim1', 'sim2', 'sim3', 'sim4', 'sim5']

OutImages = pltfn.ImagesetPlots(images, plotTitles, rdsp=2, cdsp=3, ndsp=6,
                                rescaleVals=[0, 1, 2, 3, 4, 5], fgsz=(20, 13))
OutImages.mplot()

patternfft = np.zeros(shape=pattern.shape, dtype=complex)

for sim in range(sims):
    patternfft[:, :, sim] = np.fft.fft2(pattern[:, :, sim], norm="ortho")
    # now pattern is fft

fftMeanMod = np.mean(np.abs(patternfft[:, :, 0:simsamp]), 2)
fftSDMod = np.std(np.abs(patternfft[:, :, 0:simsamp]), 2, ddof=1)
fftMeanArg = np.mean(np.angle(patternfft[:, :, 0:simsamp]), 2)
fftSDArg = np.std(np.angle(patternfft[:, :, 0:simsamp]), 2, ddof=1)

testpattern = np.copy(pattern[:, :, simsamp:sims])
testpatternData = np.copy(pattern[:, :, simsamp:sims])
testpatternNoise = np.random.normal(scale=noiseSD, size=testpatternData.shape)
testpatternData = testpatternData + testpatternNoise
testpatternDatafft = np.zeros(shape=testpatternData.shape, dtype=complex)

images = [testpatternData[:, :, 0], testpatternData[:, :, 1],
          testpatternData[:, :, 2], testpatternData[:, :, 3],
          testpatternData[:, :, 4], testpatternData[:, :, 5]]
plotTitles = ['sim0', 'sim1', 'sim2', 'sim3', 'sim4', 'sim5']

OutImages = pltfn.ImagesetPlots(images, plotTitles, rdsp=2, cdsp=3, ndsp=6,
                                rescaleVals=[0, 1, 2, 3, 4, 5], fgsz=(20, 13))
OutImages.mplot()

for sim in range(testsamp):
    testpatternDatafft[:, :, sim] = np.fft.fft2(
        testpatternData[:, :, sim], norm="ortho")

cgmrf = np.zeros(shape=testpatternData.shape, dtype=float)
reconPost0 = np.zeros(shape=testpatternData.shape, dtype=float)
reconPost1 = np.zeros(shape=testpatternData.shape, dtype=float)
reconPost2 = np.zeros(shape=testpatternData.shape, dtype=float)
reconPost3 = np.zeros(shape=testpatternData.shape, dtype=float)
reconPost4 = np.zeros(shape=testpatternData.shape, dtype=float)

for sim in range(testsamp):  # (testsamp):
    cgmrf[:, :, sim] = mcg.conjgrad(0.01, testpatternData[:, :, sim], 1000.0)

for sim in range(testsamp):
    reconPost0[:, :, sim] = pst.emppost_n(fftMeanMod, fftSDMod, np.abs(
        testpatternDatafft[:, :, sim]), np.angle(
            testpatternDatafft[:, :, sim]), knoiseSD=knoiseSD, nv=0.1,
            adim=256)

for sim in range(testsamp):
    reconPost1[:, :, sim] = pst.emppost_n(fftMeanMod, fftSDMod, np.abs(
        testpatternDatafft[:, :, sim]), np.angle(
            testpatternDatafft[:, :, sim]), knoiseSD=knoiseSD, nv=1.0,
            adim=256)

for sim in range(testsamp):
    reconPost2[:, :, sim] = pst.emppost_n(fftMeanMod, fftSDMod, np.abs(
        testpatternDatafft[:, :, sim]), np.angle(
            testpatternDatafft[:, :, sim]), knoiseSD=knoiseSD, nv=10.0,
            adim=256)

for sim in range(testsamp):
    reconPost3[:, :, sim] = pst.emppost_n(fftMeanMod, fftSDMod, np.abs(
        testpatternDatafft[:, :, sim]), np.angle(
            testpatternDatafft[:, :, sim]), knoiseSD=knoiseSD, nv=100.0,
            adim=256)


indx = 3
distval = "gauss_gauss"
bvecs = (0.0, 1.0)
expval = 2
magfimg = testpattern[:, :, indx]
kdst = bifs.kdist2D(dim1, dim2)
parfn = bifs.invxy(bvecs, kdst, exponent=expval, normimg=magfimg)

r1 = 1
r2 = 0.1
rval = 3000
# fftM = r1 * fftMeanMod + r2 * parfn
# fftS = r1 * fftSDMod + r2 * parfn
fftM = rval * fftMeanMod * parfn
fftS = rval * fftSDMod * parfn
fftM[0, 0] = fftMeanMod[0, 0]
fftS[0, 0] = fftSDMod[0, 0]

for sim in range(testsamp):
    reconPost4[:, :, sim] = pst.emppost_n(fftM, fftS, np.abs(
        testpatternDatafft[:, :, sim]), np.angle(
            testpatternDatafft[:, :, sim]), knoiseSD=knoiseSD, nv=10.0,
            adim=256)

images = [testpattern[:, :, indx], testpatternData[:, :, indx],
          cgmrf[:, :, indx], cgmrf[:, :, indx],
          reconPost0[:, :, indx], reconPost1[:, :, indx],
          reconPost2[:, :, indx], reconPost4[:, :, indx]]
plotTitles = ['simClean', 'simPlusNoise', 'cg', 'cg rescale', 'recon0.1',
              'recon1', 'recon10', 'reconProd']

OutImages = pltfn.ImagesetPlots(images, plotTitles, rdsp=2, cdsp=4, ndsp=8,
                                rescaleVals=[0, 2, 4, 5, 6, 7], fgsz=(20, 13))
OutImages.mplot()


OutDir = "ResultsImages/GalaxiesSim/"
OutPlotImages2 = pltfn.ImagesetOutPlots(images, plotTitles, outdir=OutDir,
                                        rdsp=2, cdsp=4, ndsp=8,
                                        rescaleVals=[0, 2, 4, 5, 6, 7],
                                        fgsz=(20, 13))
OutPlotImages2.mplot()
