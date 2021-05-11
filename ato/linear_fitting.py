#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import seed
from linear_data_generator import lindata_yerr_gauss
from linear_fit_utils import linear_LS_fit

seed(142)

N = 40
xmin = -4.1
xmax = 17.3
m = 0.6
b = 2.0
sigmay = 0.6

# some data to make the fit
xval, yval = lindata_yerr_gauss(N, m, b, xmin, xmax, sigmay)

# some data to show the actual line, from which the data above is
# derived
xrval = np.linspace(xmin, xmax, 3)
yrval = m*xrval + b

## Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111, title='$y=mx+b$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
# generated data
ax1.errorbar(xval, yval, 2*sigmay, None, ".g", label = 'data with $2\sigma$ errors')
ax1.errorbar(xval, yval, sigmay, None, ".b", label = 'data with $1\sigma$ errors')
# the actual line (w/o errors)
ax1.plot(xrval, yrval, "-r", label = "line with actual parameters")
# the fit
a_LSfit, b_LSfit = linear_LS_fit(xval, yval, sigmay)
x_LSfit_val = np.linspace(xmin, xmax, 3)
y_LSfit_val = a_LSfit + b_LSfit*x_LSfit_val
ax1.plot(x_LSfit_val, y_LSfit_val, "-k", label = "line with LS fit parameters")

ax1.legend(loc='upper left')
plt.show()
