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
#pytta.list_devices()

#%%
freq_min = 100
freq_max = 10000
fs = 51200
fft_degree = 18
start_margin = 0.1
stop_margin = 0.5

xt = pytta.generate.sweep(freqMin = freq_min,
  freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, startMargin = start_margin,
  stopMargin = stop_margin, method = 'logarithmic', windowing='hann')

#%% Ni measurement object

ni_meas = NIMeasurement(reference_sweep=xt, fs = fs, buffer_size = 2**8)
ni_meas.get_system_and_channels()
ni_meas.set_output_channels(out_channel_to_ni = 3, out_channel_to_amp = 1, ao_range = 10.0)
ni_meas.set_input_channels(in_channel_ref = 3, in_channel_sensor = [0],
                           ai_range = 1, sensor_sens = 50, sensor_current = 2.2e-3)
#ni_meas.initializer()
# First measurement is bad
#ni_meas.play_rec()
#%% acquire several measurements
n_repeats = 1
# Initialize
ht_mic_xt = np.zeros((2**fft_degree, n_repeats))
ht_mic_lb = np.zeros((2**fft_degree, n_repeats))

# loop through measurements
for jm in range(n_repeats):
    print('Measurement {} of {}'.format(jm+1, n_repeats))
    # Both channels in a single SignalObj
    # xt_ref, yt_mic = ni_meas.play_rec()
    meas_sig_obj = ni_meas.play_rec()
    meas_sig_obj.split()
    
    # compute h(t)    
    ht_vs_xt = pytta.ImpulsiveResponse(excitation = xt, recording = meas_sig_obj[1], samplingRate = fs, 
                                 regularization = True, freq_limits = [freq_min, freq_max])
    
    ht_ch1_ch2 = pytta.ImpulsiveResponse(excitation = meas_sig_obj[0], recording = meas_sig_obj[1], samplingRate = fs, 
                                 regularization = True, freq_limits = [freq_min, freq_max])
    
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
#ht_mic_xt_pytta.plot_time()
ht_mic_lb_pytta.plot_time() #xLim = (0, 0.006)

#%% 
plt.figure(figsize = (8, 4))
plt.plot(ht_mic_lb_pytta.timeVector, 20*np.log10(np.abs(ht_mic_lb_pytta.timeSignal)/np.amax(np.abs(ht_mic_lb_pytta.timeSignal))), alpha = 0.8)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
plt.xlim((-0.1, ht_mic_lb_pytta.timeVector[-1]))
plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()
#%%
# path = 'D:/Work/UFSM\Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/loopback_test/'
# suffix = '_fftdegree19_stopmargin1_nreps' + str(n_repeats) + '.hdf5'
# filename_htxt = 'NI34cm_ht_mic_xt' + suffix
# pytta.save(path + filename_htxt, ht_mic_xt_pytta)

# filename_htlb = 'NI34cm_ht_mic_lb' + suffix
# pytta.save(path + filename_htlb, ht_mic_lb_pytta)

# #%%
# med_dict = pytta.load(path + filename_htxt)
# keyslist = list(med_dict.keys())
# ht_mic_xt_pytta = med_dict[keyslist[0]]  