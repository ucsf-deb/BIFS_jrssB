#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:38:52 2021.

Prior/likelihood sets for various reconstructions.
@author: kornak
"""

""" Mandrill, invxy, distval = ExpRiceSq """
nsdval = 50.0  # values to use for noiseSD
panels = 6  # number of recon panels in plot
bvecs = ((0.0, 1.0), (10.0, 1.0), (5.0, 1.0), (1.0, 1.0), (0.0, 1.0),
         (0.0, 1.0))
expvals = (1.5, 2, 2, 2, 2, 2.5)
distval = "ExpRiceSq"


""" Mandrill, invxy, distval = ExpRice """
nsdval = 50.0  # values to use for noiseSD
panels = 6  # number of recon panels in plot
bvecs = ((0.0, 1.0), (10.0, 1.0), (5.0, 1.0), (1.0, 1.0), (0.0, 1.0),
         (0.0, 1.0))
expvals = (0.1, 0.5, 1, 1.5, 2, 2.5)
distval = "ExpRice"
