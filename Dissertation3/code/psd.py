import sqlalchemy as sql
import pandas as pd
import glob
from os import path, sep
import numpy as np

from numpy.fft import fft, fftshift, fftfreq, ifftshift
from collections import namedtuple

FFT = namedtuple('FTT','freq,psd_db,psd,phase'.split(','))

def psd(signal, sampling_rate,hide_bias=False):
    n = len(signal)
    windowed_signal = np.hamming(n) * signal
    g = fftshift(fft(windowed_signal))
    psd = np.abs(g)
    psd_db = 10*np.log10(psd)
    freq = fftshift(fftfreq(n, sampling_rate))
    if hide_bias:
        psd_db[int(n/2)-1:int(n/2)+2]=0
    return FFT(freq, psd_db, psd, np.angle(g))
