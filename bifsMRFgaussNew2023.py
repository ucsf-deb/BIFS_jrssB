#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:43:39 2020.

BIFS for simulated brain image
@author: kornak
"""

import numpy as np
import matplotlib.pyplot as plt
import simMRIfns as smr
import fitfns as fit
import plotfns as pltfn
import mrfconjgrad as mcg
import bifsfns as bifs

""" CONSTANTS """

ADIM = 128  # x and y dimensions of brain image matrix.
NSAMPS = 1000  # Number of samples of GMRF used.

np.set_printoptions(precision=3)
OutDir = "ResultsImages/MRFbrainSim/"

""" MAIN BODY OF CODE """

""" Generate simulated brain data. """
(noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, magfimgF,
 argfimgF, knoiseSDest, gmsk, wmsk) = smr.simbrainMRI(adim=ADIM, sdi=0.25)

# (noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, magfimgF,
#  argfimgF, knoiseSDest) = smr.simFreq(adim=ADIM, sdi=1.0)

""" GMRF Conjugate Gradients analysis. """
conjGrecon = mcg.conjgrad(noiseSD, imgPlusNoise, itrns=100, itau2=1.0)
# was 0.01 previously

""" GMRF Iterated Conditional Modes analysis. """

icmrecon = mcg.icmGauss(noiseSD, imgPlusNoise, kappa=1.0, itrns=100)
# icmrecon = mcg.icmMedian(noiseSD, imgPlusNoise, kappa=1.0, itrns=10)
# also kappa = 10.0 for median
# icmrecon = mcg.icmEdge1(noiseSD, imgPlusNoise, kappa=1.0, delta=0.1,
#                         itrns=10)

""" Reading Gaussian MRF simulations and manipulationsq. """

# Reading Gaussian MRF simulations and manipulations
# gmrfl = ("/Users/kornak/Documents/Projects/KspacePrior/"
#          "MatlabCode/MatlabAnalysis/GMRFsim/GMRFsimDataAndCode/"
#          "sampskappa_100.0.dat")  # 100.0

gmrfl = ("/Users/kornak/Documents/Projects/KspacePrior/"
         "RinlaGMRFsim/iq1gmrf.dat")  # 100.0 iq4gmrfAdjust

(gmrfmat, gmrfsd, gmrfKmeans, gmrfKsds, gmrfFFT, gmrfModFFT, gmrfModFFTsq,
 gmrfMeanModFFTsq, gmrfSDevModFFTsq) = smr.manipGmrfSim(
     cleanImage, gmrfl, adim=ADIM, nsamps=NSAMPS)

##############################################################################

""" Fit Exp/inv-poly model to mean of modulus squared vs. dist. """

sv2 = (0.0001, 0.0, 0.0, 0.0)
(qExpParEstSq, qExpPredSq) = fit.invQmodelfit(kdst, gmrfMeanModFFTsq,
                                              qstartPars=sv2)

fig, ax = plt.subplots()
plt.plot(kdst, gmrfMeanModFFTsq, 'bo', markersize=2)
plt.plot(kdst[1:], qExpPredSq[1:], 'r-', linewidth=0.5)
plt.title('Exponential prior / inv polynomial par fn for modulus squared')
ax.set(xlabel='kdst', ylabel='FS Modulus squared')
plt.show()

np.mean(np.abs(qExpPredSq[1:] - gmrfMeanModFFTsq[1:]))
np.mean(qExpPredSq[1:] - gmrfMeanModFFTsq[1:])

# Square root plot for modulus

fig, ax = plt.subplots()
plt.plot(kdst, np.sqrt(gmrfMeanModFFTsq), 'bo', markersize=1.5,
         label="Sample mean")
plt.plot(kdst[1:], np.sqrt(qExpPredSq[1:]), 'r-', linewidth=0.5,
         label="Fitted parameter function")
ax.set(xlabel='Distance from origin', ylabel='Modulus')
plt.legend(loc="upper right", markerscale=2)
plt.savefig(OutDir + 'Modulus' + ".pdf")
plt.show()

"""
# For testing ###########################################################
lngth = np.shape(qExpPredSq)[0] - 1
qExpPredSq[1:] = qExpPredSq[1:] - 0.45 * np.ones(lngth)

fig, ax = plt.subplots()
plt.plot(kdst, gmrfMeanModFFTsq, 'bo', markersize=2)
plt.ylim(0, 100)
plt.plot(kdst[1:], qExpPredSq[1:], 'r-', linewidth=0.5)
plt.title('Exponential prior / inv polynomial par fn for modulus squared')
ax.set(xlabel='kdst', ylabel='FS Modulus squared')
plt.show()
"""

##############################################################################

""" GMRF: To simulate data based only on independent prior assumption. """
""" Note that the below only simulates random noise. """
# Generating random noise simulations with same sd as gmrf.
np.random.seed(1)
iidGaussRandomMat = gmrfsd * np.reshape(
    np.random.normal(0.0, gmrfsd, cleanImage.size), cleanImage.shape)


""" Simulate actual BIFS GMRF approximations from K-space data and compare. """

np.random.seed(10)
exp2FSsim = fit.simMRFbifs(qExpPredSq, qExpPredSq, adim=ADIM, nsamps=NSAMPS,
                           dist="exp", square=True)

# exp2FSsim = fit.simMRFbifs(linPred, linPred, adim=ADIM, nsamps=NSAMPS,
#                            dist="exp", square=True)

""" Generate ACF for each of GMRF and BIFS GMRFapprox. """

# For each image take the FFT

(gmrf, gmrfACF) = fit.toEstACF(gmrfmat)
(exp2bif, exp2ACF) = fit.toEstACF(exp2FSsim)

fig, ax = plt.subplots()
plt.plot(kdst, gmrfACF, 'bo', markersize=0.2, label="GMRF")
plt.plot(kdst, exp2ACF, 'ro', markersize=0.1, label="BIFS")
plt.title('ACFs')
plt.legend(loc="upper right", markerscale=20)
ax.set(xlabel='kdst', ylabel='FS Modulus')
plt.show()

fig, ax = plt.subplots()
# plt.plot(kdst, exp2ACF, 'go', markersize=0.7, label="exp")
plt.plot(kdst, gmrfACF, 'bo', markersize=1.0, label="GMRF")
plt.plot(kdst, exp2ACF, 'ro', markersize=1.0, label="BIFS")
plt.title('ACFs')
plt.legend(loc="upper right", markerscale=2)
ax.set(xlabel='kdst', ylabel='ACF')
plt.savefig(OutDir + 'ACF' + ".pdf")
plt.show()


simImages = [gmrfmat[0, :, :], gmrfmat[1, :, :], gmrfmat[2, :, :],
             exp2FSsim[0, :, :], exp2FSsim[1, :, :], exp2FSsim[2, :, :]]
simPlotTitles = ['gmrf0', 'gmrf1', 'gmrf2',
                 'bifs0', 'bifs1', 'bifs2']

OutImagesSim = pltfn.Imageset(simImages, simPlotTitles, rdsp=2, cdsp=3, ndsp=9,
                              fgsz=(10, 7), rescale=True)
# OutImagesSim.mplot()

OutImagesS = pltfn.ImagesetOutPlots(simImages, simPlotTitles, outdir=OutDir,
                                    rdsp=2, cdsp=3, ndsp=6,
                                    rescaleVals=[0, 1, 2, 3, 4, 5],
                                    fgsz=(20, 13))
OutImagesS.mplot()


""" NOW LOOKING AT APPLYING THESE PRIORS TO THE BRAIN SIM """

""" Posterior calculations and maps. """

GMRFSC = 1.0

gmrfKmeans.shape = (ADIM, ADIM)
gmrfKsds.shape = (ADIM, ADIM)
gmrfMeanModFFTsq.shape = (ADIM, ADIM)

""" GMRF model A: empirical k-space estimates. """
gmrfEmpRec = bifs.bifs_post_mode(magfimgF, argfimgF, knoiseSD,
                                 meanfn=gmrfMeanModFFTsq, dist="expsq_rice")
# gmrfEmpiricalRecon = bifs.bifs_post_modemppost(
#    gmrfKmeans, gmrfKsds, gmrfSDparEst, magfimgF, argfimgF, knoiseSD,
#    adim=ADIM, gmrfPriorScale=GMRFSC)

""" GMRF model B: model k-space (exponential of square term model) """
gmrfModel = bifs.bifs_post_mode(magfimgF, argfimgF, knoiseSD,
                                meanfn=qExpPredSq, dist="expsq_rice")

# gmrfExpSqRecon = pst.exp2post(gmrfPred, gmrfFittedSD, magfimgF, argfimgF,
#                              knoiseSD, adim=ADIM, gmrfPriorScale=GMRFSC)


konst = 1.0  # 8.0
gmn = konst * gmrfKmeans
gsd = konst * gmrfKsds

""" GMRF model C: model k-space (empirical Gauss-Gauss approximation) """
gmrfEmpGauss = bifs.bifs_post_mode(magfimgF, argfimgF, knoiseSD,
                                   meanfn=gmn, sdfn=gsd,
                                   dist="gauss_gauss")


""" test """
# knoiseSD = 100


"""Plots. """

# Preparing for images

# l1mrfmat2 = l1mrfmat/20

images = [cleanImage, imgPlusNoise, conjGrecon, icmrecon, gmrfModel,
          gmrfEmpRec]
# images = [cleanImage, cleanImage, conjGrecon, icmrecon, gmrfEmpRec,
#           gmrfEmp2]

plotTitles = ['Truth', 'Truth + noise', 'CG - GMRF',
              'ICM recon', 'BIFS ~ GMRF par expsq',
              'BIFS ~ GMRF Emp rec expsq']

OutImages = pltfn.Imageset(images, plotTitles, rdsp=2, cdsp=3, ndsp=9,
                           fgsz=(10, 7), rescale=False)
OutImages.mplot()

cgresid = cleanImage - conjGrecon
icmresid = cleanImage - icmrecon
geresid = cleanImage - gmrfEmpRec
gmresid = cleanImage - gmrfModel

images2 = [cgresid, icmresid, gmresid]
plotTitles2 = ['CG resid', 'ICM resid', 'model GMRF resid']
OutImages2 = pltfn.Imageset(images2, plotTitles2, rdsp=1, cdsp=3, ndsp=9,
                            fgsz=(10, 4), rescale=True)
OutImages2.mplot()


np.sum(gmrfmat ** 2)/NSAMPS
np.real(np.sum(gmrfFFT * np.conj(gmrfFFT)) / NSAMPS)/(ADIM ** 2)

np.sum(np.sqrt(gmrfmat ** 2))/NSAMPS
np.real(np.sum(np.sqrt(gmrfFFT * np.conj(gmrfFFT)))) / NSAMPS
np.real(np.sum(np.sqrt(gmrfFFT * np.conj(gmrfFFT)))) / (NSAMPS * ADIM)


def printrange2(x, factor=1):
    print(factor * np.min(x), factor * np.max(x))


def printrange(x, factor=1):
    print(factor * np.mean(x), factor * np.std(x))


def printssres(x):
    print(np.sum((cleanImage - x) ** 2))


def printssmeanres(x):
    print(np.mean((cleanImage - x) ** 2))


def printrootssmeanres(x):
    print(np.sqrt(np.mean((cleanImage - x) ** 2)))


def printss(x):
    print(np.sum(x ** 2))


printrange(cleanImage)
printrange(imgPlusNoise)
printrange(conjGrecon)
printrange(icmrecon)
printrange(gmrfEmpRec)
printrange(gmrfModel)
printrange(gmrfEmpGauss)

printssres(conjGrecon)
printssres(icmrecon)
printssres(gmrfEmpRec)
printssres(gmrfModel)
printssres(imgPlusNoise)

printssmeanres(conjGrecon)
printssmeanres(icmrecon)
printssmeanres(gmrfEmpRec)
printssmeanres(gmrfModel)
printssmeanres(imgPlusNoise)

printrootssmeanres(conjGrecon)
printrootssmeanres(icmrecon)
printrootssmeanres(gmrfEmpRec)
printrootssmeanres(gmrfModel)
printrootssmeanres(imgPlusNoise)

# printss(conjGrecon - gmrfModel)
# printss(cleanImage - imgPlusNoise)

np.sum(gmsk * cleanImage)/np.sum(gmsk)
np.sum(wmsk * cleanImage)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * cleanImage)/np.sum(1 - (gmsk + wmsk))

np.sum(gmsk * imgPlusNoise)/np.sum(gmsk)
np.sum(wmsk * imgPlusNoise)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * imgPlusNoise)/np.sum(1 - (gmsk + wmsk))

np.sum(gmsk * conjGrecon)/np.sum(gmsk)
np.sum(wmsk * conjGrecon)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * conjGrecon)/np.sum(1 - (gmsk + wmsk))

np.sum(gmsk * icmrecon)/np.sum(gmsk)
np.sum(wmsk * icmrecon)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * icmrecon)/np.sum(1 - (gmsk + wmsk))

np.sum(gmsk * gmrfModel)/np.sum(gmsk)
np.sum(wmsk * gmrfModel)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * gmrfModel)/np.sum(1 - (gmsk + wmsk))

np.sum(gmsk * gmrfEmpRec)/np.sum(gmsk)
np.sum(wmsk * gmrfEmpRec)/np.sum(wmsk)
np.sum((1 - (gmsk + wmsk)) * gmrfEmpRec)/np.sum(1 - (gmsk + wmsk))

imgRecon = []

images = [cleanImage, imgPlusNoise, conjGrecon, icmrecon, gmrfModel,
          gmrfEmpRec]

plotTitles3 = ['cleanImage', 'imgPlusNoise', 'conjGrecon', 'icmrecon',
               'gmrfModelExp2', 'gmrfEmpExp2']

plotTitles2 = ['', '', '', '', '', '']

OutImages = pltfn.ImagesetPlots(images, plotTitles3, rdsp=2, cdsp=3, ndsp=6,
                                rescaleVals=[0, 1, 2, 3, 4, 5], fgsz=(20, 13))
OutImages.mplot()

OutPlotImages = pltfn.ImagesetOutPlots(images, plotTitles3, outdir=OutDir,
                                       rdsp=2, cdsp=3, ndsp=6,
                                       rescaleVals=[0, 1, 2, 3, 4, 5],
                                       fgsz=(20, 13))
OutPlotImages.mplot()

Images2 = [cgresid, icmresid, gmresid]
plotTitles4 = ['CGresid', 'ICMresid', 'modelGMRFresid']
OutImages2 = pltfn.Imageset(images2, plotTitles2, rdsp=1, cdsp=3, ndsp=9,
                            fgsz=(10, 4), rescale=True)

OutImages2.mplot()
OutPlotImages2 = pltfn.ImagesetOutPlots(images2, plotTitles4, outdir=OutDir,
                                        rdsp=1, cdsp=3, ndsp=3,
                                        rescaleVals=[0, 1, 2],
                                        fgsz=(10, 13))
OutPlotImages2.mplot()
