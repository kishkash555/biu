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
    while ii <= len(xx)-1:
        if i < len(pc_signal_x)-1 and xx[ii] >= pc_signal_x[i+1]:
            i += 1
        yy[ii] = pc_signal_y[i]
        ii += 1
    return xx, yy


def pw_constant_convolve(pc_signal_x, pc_signal_y, pc_signal_stop, h, h_width, h_min_width, xx):
    # pc_signal_x: the *start point* of the segments with constant values
    # pc_signal_y: the constant values
    # principle: 
    # - maintain which starting points are currently relevant
    # - shift the filter and use partial cumulative sums

    yy = np.nan*np.ones_like(xx)
    seg_start = seg_end = 0
    max_seg_start = len(pc_signal_x)-2
    cum_h = np.cumsum(h)
    for i,x in enumerate(xx):
        if pc_signal_x[seg_start] > x - h_min_width:
            continue # this point cannot be caluclated
        if pc_signal_stop < x + h_min_width:
            break # any further point is out of scope
        # check if the start point is obsolete:
        while (seg_start <= max_seg_start and
            pc_signal_x[seg_start+1] <= x - h_width):
            seg_start += 1

        # check if the end point needs to be extended:
        while (seg_end <= max_seg_start 
            and pc_signal_x[seg_end+1] < x + h_width):
            seg_end += 1

        rel_sig_x = pc_signal_x[seg_start:seg_end+1] - x
        rel_sig_y = pc_signal_y[seg_start:seg_end+1]

        yy[i] = pw_sumproduct(rel_sig_x,rel_sig_y,cum_h,h_width, pc_signal_stop)
    return yy


def pw_sumproduct(sig_x, sig_y, cum_h, h_width, pc_signal_stop):
    h_period = h_width / np.floor(len(cum_h)/2)
    
    # h is even: h goes from -h_width
    sig_x_as_h_index = (sig_x + h_width)/h_period
    sig_x_as_h_index = np.around(sig_x_as_h_index,3).astype(int)
    sig_stop_as_h_index = (pc_signal_stop + h_width)/h_period
    sig_stop_as_h_index = np.around(sig_stop_as_h_index,3).astype(int)

    add_initial_ele = sig_x_as_h_index[0] < 1 
    if add_initial_ele:
        sig_x_as_h_index[0] = 1
    
    if len(sig_x_as_h_index) >=2:
        pw_h = cum_h[sig_x_as_h_index[1:]-1] - cum_h[sig_x_as_h_index[:-1]-1] 
        pw_h[0] += cum_h[0]*add_initial_ele
        contrib_by_segment = pw_h*sig_y[:-1]
    else:
        pw_h = np.array([cum_h[0]*add_initial_ele])
        contrib_by_segment = pw_h * sig_y[0]
        
    contrib_last_segment = sig_y[-1] * (
        cum_h[min(len(cum_h)-1, sig_stop_as_h_index)] - cum_h[sig_x_as_h_index[-1]-1]
        )
    total = contrib_by_segment.sum() + contrib_last_segment

    return total


