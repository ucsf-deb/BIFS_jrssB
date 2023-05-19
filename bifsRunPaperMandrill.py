#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:45:44 2021.

BIFS for FS defined priors on arbitrary images
@author: kornak
"""

from PIL import Image
import numpy as np
import bifsfns as bifs
import plotfns as pltfn


nsdval = 75.0  # values to use for noiseSD
panels = 6  # number of recon panels in plot
bvecs = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
         (0.0, 1.0))
expvals = (1.5, 1.75, 1.9, 2, 2.5, 3)
distvals = ("gauss_gauss", "exp_gauss", "gauss_rice", "exp_rice", "expsq_rice")
distval = distvals[3]

# imgfl = "ExampleImages/Downloaded/MoonSurface.tiff"
imgfl = "ExampleImages/Downloaded/mandril_gray.tif"
# imgfl = "ExampleImages/Downloaded/lena_gray_512.tif"

cleanimg = Image.open(imgfl).convert('L')

arr = np.array(cleanimg.getdata(), dtype=np.uint8)
field = np.resize(arr, (cleanimg.size[1], cleanimg.size[0]))
out = field
cleanimg = Image.fromarray(out, mode='L')
# cleanimg.show()

imgarr = field.astype(np.float64)

noisyimg = Image.open(
    "ExampleImages/Downloaded/kinda_noisy_mandril_gray.tif").convert('L')
narr = np.array(noisyimg.getdata(), dtype=np.uint8)
nfield = np.resize(narr, (noisyimg.size[1], noisyimg.size[0]))

nout = nfield
noisyimg = Image.fromarray(nout, mode='L')
# noisyimg.show()

nimgarr = nfield.astype(np.float64)

"""
(knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg,
 argfimg, logknoiseSD, knoiseMean, logknoiseMean) = bifs.genFSdataTdist(
     imgarr, noiseSD=nsdval, tdf=10)
"""

(knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg,
 argfimg, logknoiseSD, knoiseMean, logknoiseMean) = bifs.genFSdata(
     imgarr, noiseSD=nsdval)

parfn = []
for i in range(panels):
    parfn.append(bifs.invxy(bvecs[i], kdst, exponent=expvals[i],
                            normimg=magfimg))

"""
parfn.append(bifs.torus(kdst, 10, 1000, normimg=magfimg))
parfn.append(bifs.invxy(bvecs[i], kdst, exponent=expvals[i],
                            normimg=magfimg))
parfn.append(bifs.invxy(bvecs[i], kdst, exponent=expvals[i],
                            normimg=magfimg) * bifs.torus(kdst, 20, 1000))
parfn.append(0.1 * bifs.invxy(bvecs[i], kdst, exponent=expvals[i],
                                  normimg=magfimg) + 20.0 * bifs.invxy(
                                  bvecs[i], kdst, exponent=expvals[i],
                                  normimg=magfimg) * bifs.torus(kdst, 1,
                                                                4))

plotTitles = ['Truth', 'Noise', 'Truth + noise',
              'Recon1', 'Recon2', 'Recon3',
              'Recon4', 'Recon5', 'Recon6']
"""

imgRecon = []
for i in range(panels):
    imgRecon.append(bifs.bifs_post_mode(magfimg, argfimg, knoiseSD, parfn[i],
                                        parfn[i],
                                        dist=distval))

imgindxs = (0, 1, 3, 4)

images = [cleanImage, imgPlusNoise, imgRecon[imgindxs[0]],
          imgRecon[imgindxs[1]], imgRecon[imgindxs[2]], imgRecon[imgindxs[3]]]

plotTitles = ['', '',
              '', '', '', '']

OutImages = pltfn.Imageset(images, plotTitles, rdsp=2, cdsp=3, ndsp=6,
                           fgsz=(20, 13), rescale=False)
OutImages.mplot()

outdir = "ResultsImages/Mandrill/t10df/"
# outdir = "ResultsImages/Mandrill/t10df/"

pltfn.plotset(images, outdir)
