# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:20:41 2025

@author: Eric Brandao

Script to measure IR with NI with sound-card playback
"""
# import sys
# sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
import matplotlib.pyplot as plt
from ni_measurement import NIMeasurement
import pytta
#%% List Sound card devices
pytta.list_devices()

#%% Generate a reference sweep
fs_sc = 44100
xt = pytta.generate.sweep(freqMin = 100, freqMax = 10000, samplingRate = fs_sc,
  fftDegree = 18, startMargin = 0.1, stopMargin = 0.5, method = 'logarithmic', windowing='hann')

print("Time length of the sweep is {}".format(xt.timeVector[-1]))
#%% Ni measurement object
fs_ni = 51200
ni_meas = NIMeasurement(time_length = 7, reference_signal = None, 
                        fs = fs_ni, buffer_size = 2**10)
ni_meas.get_system_and_channels()
ni_meas.set_sensor_properties(sensor_type = 'voltage', physical_channel_num = 0, 
                              sensitivity = 1, ai_range = 5)
ni_meas.set_sensor_properties(sensor_type = 'microphone', physical_channel_num = 1,
                          sensor_current = 2e-3, sensitivity = 45.8, ai_range = 130)

#%%
yt = ni_meas.sc_play_rec(reference_signal = xt, device = 16)

#%%
yt_list = yt.split()
ht = pytta.ImpulsiveResponse(excitation = yt_list[0], recording = yt_list[1], 
                             samplingRate = ni_meas.fs,
                             regularization = True, freq_limits = [100, 10000])

ht.IR.plot_time(xLim = (0, 0.4))
ht.IR.plot_time_dB(yLim = (-125, -45))
ht.IR.plot_freq(yLim = (-140, -15));