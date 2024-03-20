# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:23:05 2022 - minimum measurement script
"""
import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
#%%
name = 'testing_meas'
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
# arduino_dict = dict()
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 44100, fft_degree = 19, start_stop_margin = [0.1, 0.5], 
    mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7])
#%%
meas_obj.pytta_set_device(device = 14)
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 1,
                       freq_max = None, n_zeros_pad = 0)
meas_obj.pytta_play_rec_setup()
#%%
receiver_obj = Receiver()
# receiver_obj.double_planar_array(x_len=0.1,n_x=2,y_len=0.1,n_y=2, zr=0.015, dz=0.03)
receiver_obj.double_rec(z_dist = 0.02)
pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"
meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)
#%%
meas_obj.set_motors()
#%%
yt_list = meas_obj.sequential_measurement(meas_with_ni = False,
      repetitions = 2)