# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:23:05 2022 - minimum measurement script
"""
# import sys
# sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%%

# name = 'testing_meas'
# main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
name = 'pcc_tests' #'melamine_L60cm_d3cm_s100cm_2mics_17072024' # Remember good practices --> samplename_arraykeyword_ddmmaaaa
main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/'# use forward slash

# arduino_dict = dict()
source = Source(coord = [0, 0, 1.0])
#%%
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 44100, fft_degree = 18, start_stop_margin = [0.1, 0.5], 
    mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7], 
    material = None, material_type = 'pcc_tests',
    temperature = 20, humidity = 0.5,
    microphone_type = 'Behringer ECM 8000',
    audio_interface = 'Scarlet 4i4 4th Gen',
    amplifier = 'BK 2718',
    source_type = 'spherical speaker', source = source,
    start_new_measurement = True)

#%%
meas_obj.set_measurement_date()

#%%
meas_obj.pytta_list_devices()
#%%
meas_obj.pytta_set_device(device = 16)
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
                       freq_max = 10000, n_zeros_pad = 0)

#%%
meas_obj.pytta_play_rec_setup(in_channel = [1, 3], out_channel = [1, 2],
                         in_channel_ref = 3, in_channel_sensor = 1,
                         output_amplification = -3, repetitions = 2)

#%%
yt = meas_obj.pytta_play_rec()
#%%
ht = meas_obj.ir(yt, regularization=True, deconv_with_rec = False)
ht.IR.plot_time(xLim = (0, 1000e-3))
ht.IR.plot_freq()
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
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name)
meas_obj.load()
#%%
meas_obj.set_motors()
#%%
yt_list = meas_obj.sequential_measurement(bypass_scanner = True, noise_at_each_nth = 2,
                                          pcc_min = 0.9999,
                                          max_num_of_trials = 20)

#%% load one meas and check
path = main_folder + '/' + name + '/measured_signals/' #+ '/rec0_m0.hdf5'

med_dict = pytta.load(path + 'rec0_m0.hdf5')
keyslist = list(med_dict.keys())
yts = med_dict[keyslist[0]]
yts.plot_freq(xLim = (20, 20000))