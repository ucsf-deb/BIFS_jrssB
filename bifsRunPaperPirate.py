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


nsdval = 150.0  # 75.0  # values to use for noiseSD
panels = 6  # number of recon panels in plot
bvecs = ((0.0, 1.0), (10.0, 1.0), (5.0, 1.0), (1.0, 1.0), (0.0, 1.0),
         (0.0, 1.0))
expvals = (1.5, 1.75, 1.9, 2, 2.5, 3)  # (1.5, 2, 2, 2, 2, 2.5)
hfreq = (20, 30, 40, 60, 80, 100)
hfreq2 = (30, 60, 100)
distvals = ("gauss_gauss", "exp_gauss", "gauss_rice", "exp_rice", "expsq_rice")
distval = distvals[3]

# imgfl = "ExampleImages/Downloaded/MoonSurface.tiff"
# imgfl = "ExampleImages/Downloaded/mandril_gray.tif"
# imgfl = "ExampleImages/Downloaded/lena_gray_512.tif"
imgfl = "ExampleImages/Downloaded/pirate.tiff"

cleanimg = Image.open(imgfl).convert('L')

arr = np.array(cleanimg.getdata(), dtype=np.uint8)
field = np.resize(arr, (cleanimg.size[1], cleanimg.size[0]))
out = field
cleanimg = Image.fromarray(out, mode='L')
# cleanimg.show()

imgarr = field.astype(np.float64)

(knoiseSD, noise, imgPlusNoise, cleanImage, kdst, invkdst, magfimg,
 argfimg, logknoiseSD, knoiseMean, logknoiseMean) = bifs.genFSdata(
     imgarr, noiseSD=nsdval)

parfn1 = []  # for parfn hard torus
for i in range(panels):
    parfn1.append(bifs.torus(kdst, 10, hfreq[i], normimg=magfimg))

parfn2 = []  # for parfn gauss smoothed torus
for i in range(panels):
    parfn2.append(bifs.torus_gauss(kdst, 10, hfreq[i], 1.5, normimg=magfimg))

parfn3 = []  # for parfn hard torus and Gauss smoothed torus comparison
for i in range(int(panels/2)):
    parfn3.append(bifs.torus(kdst, 10, hfreq2[i], normimg=magfimg))
for i in range(int(panels/2)):
    parfn3.append(bifs.torus_gauss(kdst, 10, hfreq2[i], 1.5, normimg=magfimg))

parfn4 = []  # for parfn gauss smoothed torus
for i in range(panels):
    parfn4.append(bifs.torus_gauss(kdst, 10, hfreq[i], 2.5, normimg=magfimg))

parfn5 = []  # for regular inv dist
for i in range(panels):
    parfn5.append(bifs.invxy(bvecs[i], kdst, exponent=expvals[i],
                             normimg=magfimg))

ixyval = 3
mi7 = 0.1
mt7 = 20.0
mi8 = np.array([0.1, 0.1, 0.1])
mt8 = 1.0 - mi8

l1 = 10.01  # 1
h1 = 50  # 5
l2 = 15.01  # 5.01
h2 = 50  # 15
l3 = 20.01  # 15.01
h3 = 50  # 30
l4 = 20.01  # 1
h4 = 30  # 5
l5 = 20.01  # 5.01
h5 = 100  # 15
l6 = 20.01  # 15.01
h6 = 200  # 30

deNsPfn = bifs.invxy(bvecs[i], kdst, exponent=expvals[ixyval], normimg=magfimg)

parfn6 = []
parfn6.append(bifs.torus(kdst, l1, h1, normimg=magfimg))
parfn6.append(bifs.torus(kdst, l2, h2, normimg=magfimg))
parfn6.append(bifs.torus(kdst, l3, h3, normimg=magfimg))
parfn6.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus(kdst, l1, h1))
parfn6.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus(kdst, l2, h2))
parfn6.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus(kdst, l3, h3))

wdth = 2.5

parfn7 = []
parfn7.append(bifs.torus_gauss(kdst, l1, h1, wdth, normimg=magfimg))
parfn7.append(bifs.torus_gauss(kdst, l2, h2, wdth, normimg=magfimg))
parfn7.append(bifs.torus_gauss(kdst, l3, h3, wdth, normimg=magfimg))
parfn7.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus_gauss(kdst, l1,
                                                                   h1, wdth))
parfn7.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus_gauss(kdst, l2,
                                                                   h2, wdth))
parfn7.append(mi7 * bifs.invxy(bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) + mt7 * bifs.invxy(
                               bvecs[ixyval], kdst, exponent=expvals[ixyval],
                               normimg=magfimg) * bifs.torus_gauss(kdst, l3,
                                                                   h3, wdth))

parfn8 = []
parfn8.append(bifs.torus_gauss(kdst, l1, h1, wdth, normimg=magfimg))
parfn8.append(bifs.torus_gauss(kdst, l2, h2, wdth, normimg=magfimg))
parfn8.append(bifs.torus_gauss(kdst, l3, h3, wdth, normimg=magfimg))
parfn8.append(mi8[0] * bifs.invxy(bvecs[ixyval], kdst,
              exponent=expvals[ixyval], normimg=magfimg) + mt8[0] *
              bifs.torus_gauss(kdst, l1, h1, wdth, normimg=magfimg))
parfn8.append(mi8[1] * bifs.invxy(bvecs[ixyval], kdst,
              exponent=expvals[ixyval], normimg=magfimg) + mt8[1] *
              bifs.torus_gauss(kdst, l2, h2, wdth, normimg=magfimg))
parfn8.append(mi8[2] * bifs.invxy(bvecs[ixyval], kdst,
              exponent=expvals[ixyval], normimg=magfimg) + mt8[2] *
              bifs.torus_gauss(kdst, l3, h3, wdth, normimg=magfimg))


parfn9 = []
parfn9.append(bifs.torus_gauss(kdst, l1, h1, wdth, normimg=magfimg))
parfn9.append(bifs.torus_gauss(kdst, l2, h2, wdth, normimg=magfimg))
parfn9.append(bifs.torus_gauss(kdst, l3, h3, wdth, normimg=magfimg))
parfn9.append(bifs.torus_gauss(kdst, l4, h4, wdth, normimg=magfimg))
parfn9.append(bifs.torus_gauss(kdst, l5, h5, wdth, normimg=magfimg))
parfn9.append(bifs.torus_gauss(kdst, l6, h6, wdth, normimg=magfimg))

parfn = parfn9

imgRecon = []
for i in range(panels):
    imgRecon.append(bifs.bifs_post_mode(magfimg, argfimg, knoiseSD, parfn[i],
                                        parfn[i], dist=distval))

denoise = bifs.bifs_post_mode(magfimg, argfimg, knoiseSD, deNsPfn, deNsPfn,
                              dist=distval)

images = [cleanImage, imgPlusNoise, denoise] + imgRecon

plotTitles = ['Truth', 'Truth + noise', 'denoise']
for i in range(panels):
    plotTitles.append('recon' + str(hfreq[i]))

parimgs = []
for i in range(panels):
    parimgs.append('parfn' + str(hfreq[i]))

parimgs.append('kdst')

parfn.append(kdst)

# OutImages2 = pltfn.Imageset(parfn, parimgs, rdsp=3,
#                             cdsp=3, ndsp=7, fgsz=(10, 10), rescale=False)
# OutImages2.mplot()

OutImages = pltfn.Imageset(images, plotTitles, rdsp=3, cdsp=3, ndsp=(3+panels),
                           fgsz=(10, 10), rescale=False)
OutImages.mplot()

outdir = "ResultsImages/Pirate/"

pltfn.plotset(images, outdir)
