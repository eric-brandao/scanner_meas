# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:49:08 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:23:05 2022 - minimum measurement script
"""
# import sys
# sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
import matplotlib.pyplot as plt
from ni_measurement import NIMeasurement

import pytta
import time

#%% Just record input
# Define NI's sampling rate
fs = 51200
# Set measurement object: inputs are: fs, buffer size (int), fft_degree or time_length [s]
ni_rec = NIMeasurement(fs = fs, buffer_size = 2**10, fft_degree = None, time_length = 3)
# You need to define a recording channel. Below is how you set a microphone
ni_rec.set_sensor_properties(sensor_type = 'microphone', physical_channel_num = 1,
                             sensor_current = 4e-3, sensitivity = 49.8, ai_range = 130)
# You can set multiple recording channels. Below is how you set a voltage channel.
ni_rec.set_sensor_properties(sensor_type = 'voltage', physical_channel_num = 0, sensitivity = 1, 
                             ai_range = 5)
# The "rec()"  method will record your data and store it on a Pytta object.
rec_signal = ni_rec.rec()
rec_signal.plot_time();

#%% Generate a sweep to play and play it
# Define NI's sampling rate
fs = 51200
# Sweep parameters
freq_min = 100
freq_max = 10000
fft_degree = 18
start_margin = 0.1
stop_margin = 0.5
# Pytta's sweep generation
xt = pytta.generate.sweep(freqMin = freq_min,
  freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, startMargin = start_margin,
  stopMargin = stop_margin, method = 'logarithmic', windowing='hann')

#%% play signal
# We will play xt. Set measurement object: inputs are: fs, buffer size (int)
ni_play = NIMeasurement(fs = 51200, buffer_size = 2**10, reference_signal = xt)
# You need to define an output channel. Below is how you set a voltage output. "physical_channel_nums" is a list because you can play the same signal through multiple outputs.
ni_play.set_output_channels(physical_channel_nums = [1], ao_range = 10)
# play it
ni_play.play()

#%% Play-rec mode
# Define NI's sampling rate
fs = 51200
# Set measurement object: inputs are: fs, buffer size (int). The reference_signal defines the number of samples and time to be played
ni_playrec = NIMeasurement(fs = fs, buffer_size = 2**10, reference_signal = xt)

# You need to define an output channel. Below is how you set a voltage output. "physical_channel_nums" is a list because you can play the same signal through multiple outputs.
ni_playrec.set_output_channels(physical_channel_nums = [1], ao_range = 10)

# You can set multiple recording channels. Below is how you set a voltage channel in channel 0.
ni_playrec.set_sensor_properties(sensor_type = 'voltage', physical_channel_num = 0, 
                                 sensitivity = 1, ai_range = 5)

# You need to define a recording channel. Below is how you set a microphone in channel 1
ni_playrec.set_sensor_properties(sensor_type = 'microphone', physical_channel_num = 1,
                          sensor_current = 4e-3, sensitivity = 49.8, ai_range = 130)

# playback and record it. Recorded signals will be stored in a Pytta object
rec_signals = ni_playrec.play_rec()
rec_signals.plot_freq()

#%% Compute Impulse Response    
xtyt = rec_signals.split()
ht = pytta.ImpulsiveResponse(excitation = xtyt[0], recording = xtyt[1], samplingRate = ni_playrec.fs, 
                             regularization = True, freq_limits=[freq_min, freq_max])
ht.IR.plot_time();