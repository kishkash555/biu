'''
Methods related to the Rayleigh test
 (e.g. Mardia & Jupp 2000,2008)
http://dx.doi.org/10.1002/9780470316979

'''

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import linregress
from scipy.signal import correlate

def RayleighTest(t, v):
    '''
    Evaluate the normalized Rayleigh test for a series of event times (t)
    at a single given frequency (v).

    Parameters
    ----------
    t : 1d array
        array of event times
    v : float
        the frequency to evaluate at

    Returns
    -------
    Float, the normalized Rayleigh test at this frequency

    '''

    n = len(t)
    theta = 2. * np.pi * v * t

    z = 1. / n * (( np.sum(np.sin(theta))**2 +
                    np.sum(np.cos(theta))**2 ))

    return z


def RayleighWithAmplitude(t, y, duration, min_T):
    # Fouries series of sparse signal
    _2pi = 2*np.pi
    freqs = np.arange(1,int(duration/min_T))/duration
    window = 0.5 - 0.5 * np.cos(_2pi*t/duration)
    z = []
    for fr in freqs:
        theta = _2pi * fr * t
        z.append( (
            (np.sum(y * window * np.sin(theta)))**2 + 
            (np.sum(y * window * np.cos(theta)))**2 
                )/len(y))
         
    return np.array(z), freqs
    

def RayleighPhase(t,v):
    n = len(t)
    theta = 2. * np.pi * v * t

    ph = np.atan2(( np.sum(np.sin(theta)),
                    np.sum(np.cos(theta)) ))

    return ph

def RayleighPowerSpectrum(times, minper=1.0, maxper=500.0, nper=100):
    '''
    Compute the power spectrum over a range of periods by evaluating the
    Rayleigh test at each frequency.

    Periods are assumed to be in units of Days. Frequencies to calculate the
    Rayleigh Test on are computed as:
    >>> freq = 1 / (per * 24 * 60 * 60)


    Parameters
    ----------
    times : 1d array
        Array of times
    minper : float, optional
        Minimum period in days to evaluate the power spectrum at.
        (Default is 1.0)
    maxper : float, optional
        Maximum period in days to evaluate the power spectrum at.
        (Default is 500.0)
    nper : int, optional
        Number of periods to evaluate the power spectrum at, linearly
        spaced from minper to maxper. (Default is 100)

    Returns
    -------
    1d float array with length of nper


    Examples
    --------
    >>> ps  = RayleighPowerSpectrum(times, minper=0.25, maxper=25, nper=123)

    '''

    #maxfreq = 1. / (minper * 24. * 60. * 60.)
    maxfreq = 1. / minper 

#    minfreq = 1. / maxper * 24. * 60. * 60.)
    minfreq = 1. / maxper 

    # Evaluate at linearly spaced frequencies
    freqs = np.linspace(minfreq, maxfreq, num=nper)

    # periods = 1. / freqs / (24. * 60. * 60.)

    # z = map(lambda v: RayleighTest(times * (24. * 60. * 60.), v), freqs)
    z = [RayleighTest(times , v) for v in freqs]

    return z, freqs

class lf_model:
    def __init__(self, t, end_time, slope, lowess_fit, lowess_res, interp_interval, interp_fit, interp_resid):
        self.t = t
        self.end_time = end_time
        self.slope = slope
        self.lowess_fit = lowess_fit
        self.lowess_res = lowess_res
        self.interp_interval = interp_interval
        self.interp_fit = interp_fit
        self.interp_res = interp_resid
    
    @classmethod
    def fit_low_frequency(clas, event_times, lowess_frac=0.2, interp_interval=0.1, model_type='avg'):
        end_time = event_times[-1]
        event_times = event_times[:-1]
        data_ordinal = np.arange(len(event_times))+1
        if model_type=='regerss':
            lin_model = linregress(event_times,data_ordinal)
            slope, intercept = lin_model.slope, lin_model.intercept
            lin_res = data_ordinal - (slope*event_times + intercept)
        
        else:
            slope = (len(event_times)-1)/(event_times[-1]-event_times[0])
            lin_res = data_ordinal - (slope*event_times)
            intercept = np.mean(lin_res)
            lin_res = lin_res -intercept        
        lowess_fit = lowess(lin_res,event_times, lowess_frac, it=0, is_sorted=True, return_sorted=False)
        lowess_res = lin_res - lowess_fit
        interp_times = np.arange(0, end_time, interp_interval)
        interp_fit = np.interp(interp_times, event_times, lowess_fit)
        interp_res = np.interp(interp_times, event_times, lowess_res)
        return clas(event_times, end_time, slope, lowess_fit, lowess_res, interp_interval, interp_fit, interp_res)


def cross_corr(self, other, max_shift, signal_to_use = 'fit'):
    attr = 'interp_fit' if signal_to_use == 'fit' else 'interp_res'
    #x1 = getattr(self, attr)
    #x2 = getattr(other, attr)
    x1 = self.interp_fit + self.interp_res
    x2 = other.interp_fit + other.interp_res
    
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    corr = correlate(x1,x2)/correlate(np.ones(len(x1)),np.ones(len(x2)))
    lx = len(x1)
    #dt = lambda n: (n-lx+1)*self.interp_interval
    dt_1 = lambda t: int(t/self.interp_interval)+lx-1
    x = np.arange(-max_shift, max_shift, self.interp_interval)
    
    y = corr[dt_1(-max_shift): dt_1(max_shift)].copy()
    peak = x[np.argmax(y)]
    return x, y, peak

def multi_cross_corr(lf_list, max_shift):
    res = []
    pairs = []
    for i in range(len(lf_list)):
        for j in range(i+1,len(lf_list)):
            res.append(cross_corr(lf_list[i],lf_list[j],max_shift))
            pairs.append((i,j))
    return res, pairs

