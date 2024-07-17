# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:45:32 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:07:17 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from ni_measurement import NIMeasurement

import pytta
import time

#%% 
pytta.list_devices()

#%%
freq_min = 100
freq_max = 10000
fs = 51200
fft_degree = 19
start_margin = 0.1
stop_margin = 1

xt = pytta.generate.sweep(freqMin = freq_min,
  freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, startMargin = start_margin,
  stopMargin = stop_margin, method = 'logarithmic', windowing='hann')

#%% Ni measurement object

ni_meas = NIMeasurement(reference_sweep=xt, fs = fs, buffer_size = 2**10)
ni_meas.get_system_and_channels()
ni_meas.set_output_channels(out_channel_to_ni = 3, out_channel_to_amp = 0, ao_range = 10.0)
ni_meas.set_input_channels(in_channel_ref = 0, in_channel_sensor = [1],
                           ai_range = 1.0, sensor_sens = 50, sensor_current = 2.2e-3)

#%% acquire several measurements
n_repeats = 10
# Initialize
ht_mic_xt = np.zeros((2**fft_degree, n_repeats))
ht_mic_lb = np.zeros((2**fft_degree, n_repeats))

# loop through measurements
for jm in range(n_repeats):
    print('Measurement {} of {}'.format(jm+1, n_repeats))
    # Both channels in a single SignalObj
    xt_ref, yt_mic = ni_meas.play_rec()
    
    
    # compute h(t)    
    ht_vs_xt = pytta.ImpulsiveResponse(excitation = xt, recording = yt_mic[0], samplingRate = fs, 
                                 regularization = True)
    
    ht_ch1_ch2 = pytta.ImpulsiveResponse(excitation = xt_ref, recording = yt_mic[0], samplingRate = fs, 
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
ht_mic_xt_pytta.plot_time()
ht_mic_lb_pytta.plot_time(xLim = (0, 0.006))

#%% 
# plt.figure()
# plt.plot(ht_mic_xt_pytta.timeSignal)
# plt.xlim((980, 1020))
# plt.grid(which = 'major');

#%%
path = 'D:/Work/UFSM\Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/loopback_test/'
suffix = '_fftdegree19_stopmargin1_nreps' + str(n_repeats) + '.hdf5'
filename_htxt = 'NI34cm_ht_mic_xt' + suffix
pytta.save(path + filename_htxt, ht_mic_xt_pytta)

filename_htlb = 'NI34cm_ht_mic_lb' + suffix
pytta.save(path + filename_htlb, ht_mic_lb_pytta)

#%%
med_dict = pytta.load(path + filename_htxt)
keyslist = list(med_dict.keys())
ht_mic_xt_pytta = med_dict[keyslist[0]]  