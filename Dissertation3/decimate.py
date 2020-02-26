import pandas as pd
import numpy as np
from numpy.fft import fft, fftshift
from scipy.signal import oaconvolve
from scipy.signal import remez


def create_lpf(n=4000,max_flatband=0.5, min_stopband=1, fs=500):
    h = remez(n+2,[0., max_flatband, min_stopband, fs/2], [1.,0.],fs=fs)
    return h[1:-1] # for some reason first and last points are bas

def pw_constant_to_dense(pc_signal_x, pc_signal_y, xx):
    # pc_signal_x: the *start point* of the segments with constant values
    # pc_signal_y: the constant values
    i = 0
    ii = 0
    yy = np.zeros(len(xx))
    while xx[ii] < pc_signal_x[i]:
        ii += 1
    xx = xx[ii:]
    yy = yy[ii:]
    ii = 0
    while i <= len(pc_signal_x)-2:
        while xx[ii+1] < pc_signal_x[i+1]:
            yy[ii] = pc_signal_y[i]
            ii += 1
        i += 1
    yy[ii:] = pc_signal_y[i]
    return yy

