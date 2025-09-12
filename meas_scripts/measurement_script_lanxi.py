# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:40:49 2024

@author: Eric Brandão
"""
import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%%
name = 'testing_meas' #'melamine_L60cm_d3cm_s100cm_2mics_17072024' # Remember good practices --> samplename_arraykeyword_ddmmaaaa
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts/'# use forward slash
# arduino_dict = dict()
source = Source(coord = [0, 0, 1.0])
#%% Measurement object
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 131072, fft_degree = 20, start_stop_margin = [0.5, 1.0],
    mic_sens = 45.8, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7], 
    material = None, material_type = 'test',
    temperature = 20, humidity = 0.5,
    microphone_type = 'BK 4189',
    audio_interface = 'LANXI 4 input / 2 output',
    amplifier = 'BK 2718',
    source_type = 'spherical speaker', source = source,
    start_new_measurement = True, sound_card_measurement = False,
    repetitions = 1)
#%%
meas_obj.set_measurement_date()
#%%
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
                       freq_max = 10000, n_zeros_pad = 0)
#%%
sensor_dict_list = [dict(sensor_type = 'microphone', physical_channel_num = 2,
                         sensitivity = 50, ai_range = 130, 
                         optimal_channel_range = False),
                    dict(sensor_type = 'voltage', physical_channel_num = 4,
                         sensitivity = 1, ai_range = 10, 
                         optimal_channel_range = False)]
meas_obj.lanxi_initializer(ip_address = "169.254.180.173", 
                           sensor_dict_list = sensor_dict_list,
                           in_channel_ref_num = 4)
#%% Usually the first measurement is bad. Look at the time signiature for mean value = 0
yt = meas_obj.lanxi_control_obj.play_rec()
yt.plot_time()
print('Mean value of recording: {}'.format(np.mean(yt.timeSignal)))
#%%
yt_list = yt.split()
ht = pytta.ImpulsiveResponse(excitation = yt_list[1], 
                             recording = yt_list[0], 
                             samplingRate = meas_obj.xt.samplingRate,
                             regularization = True, freq_limits = [100, 10000], 
                             method = 'linear')
#%%
ht = meas_obj.ir(yt, regularization=True, deconv_with_rec = True)
ht.IR.plot_time();
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
                                pcc_min = 0.999, max_num_of_trials = 2)

#%% load one meas and check
path = main_folder + '/' + name + '/measured_signals/' #+ '/rec0_m0.hdf5'

med_dict = pytta.load(path + 'rec1_m0.hdf5')
keyslist = list(med_dict.keys())
yts = med_dict[keyslist[0]]
yts.plot_freq(xLim = (20, 20000))