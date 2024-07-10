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
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%% Generate a reference sweep
fs = 51200

xt = pytta.generate.sweep(freqMin = 100, freqMax = 10000, samplingRate = fs,
  fftDegree = 18, startMargin = 0.1, stopMargin = 0.5,
  method = 'logarithmic', windowing='hann')

#%% Ni measurement object

ni_meas = NIMeasurement(reference_sweep=xt, fs = fs)
ni_meas.get_system_and_channels()
ni_meas.set_output_channels(out_channel_to_ni = 0, out_channel_to_amp = 3, ao_range = 10)
ni_meas.set_input_channels(in_channel_ref = 0, in_channel_sensor = [1],
                           ai_range = 1, sensor_sens = 45.1, sensor_current = 2.2e-3)

#%%
xt_ref, yt_mic = ni_meas.play_rec(buffer_size = 2**12)

#%%
ht = pytta.ImpulsiveResponse(excitation = xt_ref, recording = yt_mic, samplingRate = ni_meas.xt.samplingRate,
                             regularization = True, method = 'linear')

ht.IR.plot_time(xLim = (0, 0.001))
# ni_meas.play()
# name = 'testing_meas_ni'
# main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
# # arduino_dict = dict()
# source = Source(coord = [0, 0, 1.0])
# #%%
# meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
#     fs = 51200, fft_degree = 16, start_stop_margin = [0.1, 0.5], 
#     mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
#     y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 5, z_digital_pin = 28,
#     dht_pin = 40, pausing_time_array = [5, 8, 7], 
#     material = None, material_type = 'no sample',
#     temperature = 20, humidity = 0.5,
#     microphone_type = 'BK 4189',
#     audio_interface = 'NI',
#     amplifier = 'BK 2718',
#     source_type = 'random open speaker', source = source,
#     start_new_measurement = True)

# #%%
# meas_obj.set_measurement_date()

# #%%
# input_dict = dict(terminal = 'cDAQ1Mod3/ai1', mic_sens = 51.4, 
#   current_exc_sensor = 0.0022, max_min_val = [-5, 5])
# output_dict = dict(terminal = 'cDAQ1Mod4/ao3', max_min_val =  [-10,10])

# meas_obj.ni_set_config_dicts(input_dict = input_dict, output_dict = output_dict)



# #%%
# # meas_obj.pytta_set_device(device = 23)
# meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
#                        freq_max = 10000, n_zeros_pad = 0)

# #%%
# yt_obj = meas_obj.ni_play_rec()

# #%%
# ht = meas_obj.ir(yt_obj, regularization=True)
# ht.IR.plot_time(xLim = (0, 10e-3))
# # ht.IR.plot_freq()

# #%% Repetibility test
# N_meas = 5

# ht_list = []
# for imeas in range(N_meas):
#     yt_obj = meas_obj.ni_play_rec()
#     ht = meas_obj.ir(yt_obj, regularization=True)
#     ht_list.append(ht)
    
# #%% plot rep
# plt.figure()
# for imeas in range(N_meas):
#     plt.plot(ht_list[imeas].IR.timeVector, ht_list[imeas].IR.timeSignal)
# plt.xlim((0, 10e-3))

# #%%
# # meas_obj.save()

# # #%%
# # receiver_obj = Receiver(coord = [0,0,0.01])
# # receiver_obj.double_rec(z_dist = 0.02)
# # #receiver_obj.double_planar_array(x_len=0.65,n_x=5,y_len=0.57,n_y=5, zr=0.015, dz=0.03)

# # pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"
# # # This next method saves everything automaically.
# # meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)


# # #%%
# # #%%
# # meas_obj = ScannerMeasurement(main_folder = main_folder, name = name)

# # #%%
# # meas_obj.set_motors()
# # #%%
# # yt_list = meas_obj.sequential_measurement(meas_with_ni = False,
# #       repetitions = 2)