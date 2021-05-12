#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from linear_fit_utils import linear_LS_fit

a = 2.3
b = 5.5
x = np.array([1, 2, 5])
y = a + b*x
sigmay = np.array([0.2]*x.size) # this works
sigmay = 0.2 # this does not work (now it does, bug fixed)

aC, bC = linear_LS_fit(x, y, sigmay)

print("actual:", a, b)
print("calced:", aC, bC)
