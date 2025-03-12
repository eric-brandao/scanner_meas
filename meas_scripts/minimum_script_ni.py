# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:40:49 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:23:05 2022 - minimum measurement script
"""
import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%%
name = 'testing_meas_ni'
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
#%% Define your source object - coordinates are important when estimating the impedance sometimes. 
### This should be part of measurement metadata
source = Source(coord = [0, 0, 1])
#%% Instantiate your measurement object controller.
# Define NI's sampling rate
fs = 51200
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = fs, fft_degree = 18, start_stop_margin = [0.1, 0.5], 
    mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 5, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [8, 8, 8], 
    material = None, material_type = 'melamine_L60cm_d4cm',
    temperature = 20, humidity = 0.5,
    microphone_type = 'BK 4942',
    audio_interface = 'NI cDaq 9174/9263(out)/9234(in)',
    amplifier = 'BK 2718',
    source_type = 'spherical speaker', source = source,
    start_new_measurement = True, sound_card_measurement = False)

### set a date as today
meas_obj.set_measurement_date()
#%%
# meas_obj.pytta_list_devices()
#%% Set the measurement sweep. It will now save the xt in your "measured_signals" folder
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
                       freq_max = 10000, n_zeros_pad = 0)

#%% Do the NI play-rec setup. Channel numbers is super important.
meas_obj.ni_initializer(buffer_size = 2**10, repetitions = 2)
meas_obj.ni_set_output_channels(physical_channel_nums = [1], ao_range = 10)
meas_obj.ni_set_input_channels(in_channel_ref_onrec = 0, in_channel_sensor_onrec = 1,
                            ai_range = 5, sensor_sens = 49.8, sensor_current = 4e-3)
#%% Usually the first measurement is bad. Look at the time signiature for mean value = 0
yt = meas_obj.ni_play_rec()
yt.plot_time()
print('Mean value of recording: {}'.format(np.mean(yt.timeSignal)))
#%% Test one impulse response measurement
yt_xt_list = yt.split()
ht = pytta.ImpulsiveResponse(excitation = yt_xt_list[0], 
                             recording = yt_xt_list[1], 
                             samplingRate = meas_obj.xt.samplingRate,
                             regularization = True, freq_limits = [100, 10000], 
                             method = 'linear')
ht.plot_time()
#%%
ht = meas_obj.ir(yt, regularization=True, deconv_with_rec = True,
                 ref_channel = 0, recording_channel = 1)
ht.IR.plot_time()
#ht.IR.plot_freq()
#%%
# meas_obj.save()

#%%
receiver_obj = Receiver(coord = [0,0,0.01])
receiver_obj.double_rec(z_dist = 0.02)
#receiver_obj.double_planar_array(x_len=0.65,n_x=5,y_len=0.57,n_y=5, zr=0.015, dz=0.03)

pt0 = np.array([0.0, 0.0, 0.05]); "--> Coordinates where the michophone is"
# This next method saves everything automaically.
meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)


#%%
meas_obj.plot_scene(L_x = 0.6, L_y = 0.6, sample_thickness = 0.1,
               baffle_size = 1.2, elev = 30, azim = 45)
#%%
meas_obj.set_motors()
#%%
meas_obj.sequential_measurement(meas_with_ni = True, bypass_scanner = True, noise_at_each_nth = None)

#%% load one meas and check
path = main_folder + '/' + name + '/measured_signals/' #+ '/rec0_m0.hdf5'

med_dict = pytta.load(path + 'rec0_m0.hdf5')
keyslist = list(med_dict.keys())
yts = med_dict[keyslist[0]]
yts.plot_freq(xLim = (20, 20000))