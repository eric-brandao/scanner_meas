# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:07:17 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pytta
import time

#%% 
pytta.list_devices()

#%%
freq_min = 100
freq_max = 10000
fs = 44100
fft_degree = 18
start_margin = 0.1
stop_margin = 0.5

xt = pytta.generate.sweep(freqMin = freq_min,
  freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, startMargin = start_margin,
  stopMargin = stop_margin, method = 'logarithmic', windowing='hann')

#%%
pytta_meas = pytta.generate.measurement('playrec', excitation = xt,
    samplingRate = fs, freqMin = freq_min, freqMax = freq_max, device = 13,
    inChannels=[1, 2], outChannels=[1, 2], outputAmplification = -3)

#%% acquire several measurements
n_repeats = 1
# Initialize
ht_mic_xt = np.zeros((2**fft_degree, n_repeats))
ht_mic_lb = np.zeros((2**fft_degree, n_repeats))

# loop through measurements
for jm in range(n_repeats):
    print('Measurement {} of {}'.format(jm+1, n_repeats))
    # Both channels in a single SignalObj
    yt_2ch = pytta_meas.run()
    
    # Split channels to compute h(t)
    yt_mic = pytta.classes.SignalObj(signalArray = yt_2ch.timeSignal[:,0], 
        domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)
    
    yt_lb = pytta.classes.SignalObj(signalArray = yt_2ch.timeSignal[:,1], 
        domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)
    
    # compute h(t)    
    ht_vs_xt = pytta.ImpulsiveResponse(excitation = xt, recording = yt_2ch, samplingRate = fs, 
                                 regularization = True)
    
    ht_ch1_ch2 = pytta.ImpulsiveResponse(excitation = yt_lb, recording = yt_mic, samplingRate = fs, 
                                 regularization = True)
    
    # Feed numpy array
    ht_mic_xt[:, jm] = ht_vs_xt.IR.timeSignal[:,0]
    ht_mic_lb[:, jm] = ht_ch1_ch2.IR.timeSignal[:,0]
    
    # pause
    time.sleep(0.3)

#%% Feed a SignalObj with all measurements
ht_mic_xt_pytta = pytta.classes.SignalObj(signalArray = ht_mic_xt, 
    domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)

ht_mic_lb_pytta = pytta.classes.SignalObj(signalArray = ht_mic_lb, 
    domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)

#%%
ht_mic_xt_pytta.plot_time(xLim = (22e-3, 25e-3))
ht_mic_lb_pytta.plot_time(xLim = (0, 0.003))

#%% 
plt.figure()
plt.plot(ht_mic_xt_pytta.timeSignal)
plt.xlim((980, 1020))
plt.grid(which = 'major');

#%%
path = 'D:/Work/UFSM\Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/loopback_test/'
suffix = '_fftdegree19_stopmargin1_nreps' + str(n_repeats) + '.hdf5'
filename_htxt = 'ht_mic_xt' + suffix
pytta.save(path + filename_htxt, ht_mic_xt_pytta)

filename_htlb = 'ht_mic_lb' + suffix
pytta.save(path + filename_htlb, ht_mic_lb_pytta)

#%%
med_dict = pytta.load(path + filename_htxt)
keyslist = list(med_dict.keys())
ht_mic_xt_pytta = med_dict[keyslist[0]]  