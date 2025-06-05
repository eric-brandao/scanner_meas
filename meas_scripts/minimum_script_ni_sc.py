# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:20:41 2025

@author: Eric Brandao

Script to measure IR with NI with sound-card playback
"""

import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%% Folder and source
name = 'pcc_tests' #'melamine_L60cm_d3cm_s100cm_2mics_17072024' # Remember good practices --> samplename_arraykeyword_ddmmaaaa
main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/'# use forward slash
# arduino_dict = dict()
source = Source(coord = [0, 0, 1.0])
#%% Measurement object
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 51200, fft_degree = 18, 
    mic_sens = 45.8, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7], 
    material = None, material_type = 'pcc_tests',
    temperature = 20, humidity = 0.5,
    microphone_type = 'BK 4189',
    audio_interface = 'NI 4 input / Scarlet 4i4 as playback',
    amplifier = 'BK 2718',
    source_type = 'spherical speaker', source = source,
    start_new_measurement = True, sound_card_measurement = False,
    repetitions = 1)

#%% Set data
meas_obj.set_measurement_date()

#%% List Sound card devices
pytta.list_devices()
#%% We generate a reference sweep with NI's sample rate (only to compare the PCC)
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
                       freq_max = 10000, n_zeros_pad = 0)
#%% We neeed to generate a sweep for playback (with the SC sampling rate)
xt_sc = pytta.generate.sweep(freqMin = meas_obj.freq_min, freqMax = meas_obj.freq_max, 
                             samplingRate = 44100, fftDegree = 18, 
                             startMargin = 0.1, stopMargin = 1.5, 
                             method = meas_obj.method, windowing='hann')
#%%
meas_obj.ni_initializer(buffer_size = 2**10, play_rec_type = 'SC play and NI rec')
meas_obj.ni_set_output_channels() # Here, whatever - does not matter as out is SC
meas_obj.ni_set_input_channels(in_channel = [0, 1], in_channel_ref_num = 0,
                          ai_range = 5, sensor_sens = 50, sensor_current = 2e-3)
#%% Usually the first measurement is bad. Look at the time signiature for mean value = 0
yt = meas_obj.ni_control_obj.sc_play_rec(reference_signal = xt_sc, device = 16)
yt.plot_time()
print('Mean value of recording: {}'.format(np.mean(yt.timeSignal)))
#%%
yt_xt_list = yt.split()
ht = pytta.ImpulsiveResponse(excitation = yt_xt_list[0], 
                             recording = yt_xt_list[1], 
                             samplingRate = meas_obj.fs,
                             regularization = True, freq_limits = [100, 10000], )
ht.IR.plot_time(xLim = (0.035, 0.06))
#%%
ht = meas_obj.ir(yt, regularization=True, deconv_with_rec = True)
ht.IR.plot_time(xLim = (0, 2))
#ht.IR.plot_freq()
#%%
# meas_obj.save()

#%%
receiver_obj = Receiver(coord = [0,0,0.01])
receiver_obj.double_rec(z_dist = 0.02)
#receiver_obj.double_planar_array(x_len=0.65,n_x=11,y_len=0.57,n_y=10, zr=0.015, dz=0.03)
pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"
# This next method saves everything automaically.
meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)


#%%
meas_obj.plot_scene(L_x = 0.6, L_y = 0.6, sample_thickness = 0.1,
               baffle_size = 1.2, elev = 30, azim = 45)
#%%
meas_obj.set_motors()
#%%
meas_obj.sequential_measurement(bypass_scanner = True, noise_at_each_nth = 4,
                                pcc_min = 0.9999, max_num_of_trials = 2,
                                reference_signal = xt_sc, playback_device = 16)

#%% load one meas and check
path = main_folder + '/' + name + '/measured_signals/' #+ '/rec0_m0.hdf5'

med_dict = pytta.load(path + 'rec0_m0.hdf5')
keyslist = list(med_dict.keys())
yts = med_dict[keyslist[0]]
yts.plot_freq(xLim = (20, 20000))