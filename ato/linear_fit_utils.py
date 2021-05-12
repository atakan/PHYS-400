#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_stats1(x, y, sigmay):
    r'''This routine returns some "statistics" on the provided data.
    It is the most basic of its kind and returns various sums in eq.
    6.12 of Bevington and Robinson. E.g., one_s2 is
    $\sum\frac{1}{sigma^2}, xy_s2 is $\sum\frac{xy}{sigma^2} etc.
    There will be more complicated routines that will calculate
    additional statistics, hence the "1" in the name of this routine.

    sigmay is the uncertainty in y values. It can have multiple forms,
    as in matplotlib's errorbar routine's xerr or yerr arguments.
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
    
    The returned statistics are:
    one_s2, x_s2, y_s2, x2_s2, xy_s2
    '''

    if isinstance(sigmay, float):
        s2 = sigmay*sigmay
        one_s2 = x.size / s2
    elif isinstance(sigmay, np.ndarray):
        if sigmay.shape == (x.size, ) :
            s2 = sigmay*sigmay
        elif sigmay.shape == (2, x.size) :
            '''For asymmetric errors, we simply average the errors on
            two sides.'''
            # XXX this is likely not the best way
            s2 = np.mean(sigmay, axis=1)**2
        else :
            raise NotImplementedError('Only sigma of certain shapes are OK')
        one_s2 = np.sum(1.0/s2)
    else:
        raise TypeError('Only sigma of certain types are OK')

    x2 = x*x
    xy = x*y
    x_s2 = np.sum(x/s2)
    y_s2 = np.sum(y/s2)
    x2_s2 = np.sum(x2/s2)
    xy_s2 = np.sum(xy/s2)
    return one_s2, x_s2, y_s2, x2_s2, xy_s2

def linear_LS_fit(x, y, sigmay):
    '''Makes a least squares linear fit to data.
    The calculations are based on Bevington & Robinson (3rd edition),
    chapter 6. In particular, equation 6.12.
    Here we follow Bevington's notation y = a + bx, instead of
    the more common y = mx + b.
    The routine returns a, b.
    '''

    one_s2, x_s2, y_s2, x2_s2, xy_s2 = get_stats1(x, y, sigmay)
    delta = one_s2*x2_s2 - x_s2**2
    a = 1.0/delta * (x2_s2*y_s2 - x_s2*xy_s2)
    b = 1.0/delta * (one_s2*xy_s2 - x_s2*y_s2)
    return a, b
