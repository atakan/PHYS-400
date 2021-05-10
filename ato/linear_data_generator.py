#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import rand, randn

def lindata_yerr_gauss(N, m, b, xmin, xmax, sigmay):
    '''Returns linear data with gaussian errors in y.
    x values are uniform in range [xmin, xmax).
    y = mx + b + N(0, sigmay^2) where N(mu, sigma^2) is the normal
    distribution with mean mu and standard deviation sigma.
    '''
    xval = rand(N)
    xval *= xmax-xmin
    xval += xmin
    yval = m*xval + b + sigmay*randn(N)
    return xval, yval
