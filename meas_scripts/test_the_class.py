# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:14:27 2022

@author: ericb
"""
import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np


from sequential_measurement import ScannerMeasurement

# Receiver class
from receivers import Receiver

#%%
# fs_ni = 51200
name = 'testing_meas'
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
# arduino_dict = dict()
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 44100, fft_degree = 16, 
    start_stop_margin = [0.1, 0.1], mic_sens = 51.4,
    x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26,
    z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7])

#%%
#%%
input_dict = dict(terminal = 'cDAQ1Mod1/ai0', mic_sens = 51.4, 
  current_exc_sensor = 0.0022, max_min_val = [-5, 5])
output_dict = dict(terminal = 'cDAQ1Mod3/ao0', max_min_val =  [-10,10])
meas_obj.ni_set_config_dicts(input_dict = input_dict, output_dict = output_dict)

#%%
meas_obj.pytta_list_devices()

#%%
meas_obj.pytta_set_device(device = 14)
#%%

meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 1,
                       freq_max = None, n_zeros_pad = 200)

#%%
meas_obj.pytta_play_rec_setup()

#%%
yt_obj = meas_obj.pytta_play_rec() 
#%%
yt_obj = meas_obj.ni_play_rec()

#%%
ht_obj = meas_obj.ir(yt = yt_obj, regularization = True)
#%%
# meas_obj.set_arduino_parameters(x_pwm = 2, x_dig = 24,
#                                 y_pwm = 3, y_dig = 26,
#                                 z_pwm = 4, z_dig = 28,
#                                 dht = 40)
#%%
meas_obj.set_motors()

#%%
meas_obj.move_motor(motor_to_move = 'z', dist = -0.01) 
#%%
#meas_obj.set_dht_sensor()

#%%
receiver_obj = Receiver()
# receiver_obj.double_planar_array(x_len=0.1,n_x=2,y_len=0.1,n_y=2, zr=0.015, 
#                                   dz=0.03)
receiver_obj.double_rec(z_dist = 0.02)
# receiver_obj.random_3d_array(x_len = 1, y_len = 1.0, z_len = 1.0, zr = 0.1, n_total = 10, seed = 0)
# np.random.shuffle(receiver_obj.coord)


pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"

meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)



#%%
meas_obj.sequential_movement()

#%%
yt_list = meas_obj.sequential_measurement(meas_with_ni = False,
                                          repetitions = 2,
                                          plot_frf = True)
#%%
yt_list = meas_obj.load_meas_files()



