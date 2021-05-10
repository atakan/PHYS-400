#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import seed
from linear_data_generator import lindata_yerr_gauss

seed(142)

N = 40
xmin = -4.1
xmax = 17.3
m = 0.6
b = 2.0
sigmay = 0.6

xval, yval = lindata_yerr_gauss(N, m, b, xmin, xmax, sigmay)

xrval = np.linspace(xmin, xmax, 3)
yrval = m*xrval + b

## Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111, title='$y=mx+b$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
# generated data
ax1.errorbar(xval, yval, 2*sigmay, None, ".g", label = '$2\sigma$ errors')
ax1.errorbar(xval, yval, sigmay, None, ".b", label = '$1\sigma$ errors')
# the actual line (w/o errors)
ax1.plot(xrval, yrval, "-r")
# the fit (not here yet)

ax1.legend(loc='upper left')
plt.show()
