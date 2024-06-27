# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:52:45 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:23:05 2022 - minimum measurement script
"""
import numpy as np
from sequential_measurement import ScannerMeasurement
from receivers import Receiver
from sources import Source
import pytta
#%% Naming things
name = 'melamine_L60cm_d3cm_s100cm_2mics_28062024' # Remember good practices --> samplename_arraykeyword_ddmmaaaa
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash

#%% Define your source object - coordinates are important when estimating the impedance sometimes. 
### This should be part of measurement metadata
source = Source(coord = [0, 0, 1.0])
#%% Instantiate your measurement object controller.
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 44100, fft_degree = 18, start_stop_margin = [0.1, 1], 
    mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7], 
    material = None, material_type = 'melamine_L60cm_d4cm',
    temperature = 20, humidity = 0.5,
    microphone_type = 'Behringer ECM 8000',
    audio_interface = 'M-audio Fast Track Pro',
    amplifier = 'BK 2718',
    source_type = 'spherical speaker', source = source,
    start_new_measurement = True)

### set a date as today
meas_obj.set_measurement_date()

#%% List pytta devices and choose the ASIO one of your sound card
meas_obj.pytta_list_devices()
#%% Set the audio device - if input/output is separate, it should be a list [in, out]
meas_obj.pytta_set_device(device = 23)

#%% Set the measurement sweep. It will now save the xt in your "measured_signals" folder
meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 100,
                       freq_max = 10000, n_zeros_pad = 0)

#%% Do the pytta play-rec setup. Channel numbers is super important.
meas_obj.pytta_play_rec_setup(in_channel = 2, out_channel = 1, 
                         output_amplification = -3)

#%% measure loopback response and save it (if wanted)
### plug the output of the sound card on the input, measure and save the IR. 
### Serves to know the latency if desired
meas_obj.pytta_measure_loopback()

#%% You can test a measurement if you want - check for clipping and other potential problems
### If you feel like changing your sweep design, you can re-run things 
yt = meas_obj.pytta_play_rec()

#%% Chech an impulse response - Does it look ok?
ht = meas_obj.ir(yt, regularization=True)
ht.IR.plot_time()
ht.IR.plot_freq()

#%% Set your receiver array in two stages: (1) - the array; (2) - the starting point (go there and measure it)
### Good practice is that your starting point is above or below your array. 
### Make sure that the scanner can span everything you asked for.
receiver_obj = Receiver(coord = [0,0,0.01])
receiver_obj.double_rec(z_dist = 0.02)
#receiver_obj.double_planar_array(x_len=0.65,n_x=5,y_len=0.57,n_y=5, zr=0.015, dz=0.03)
starting_coordinates= np.array([0.0, 0.0, 0.04]); "--> Coordinates where the michophone is"

# Setting up the array saves every config made to this point
meas_obj.set_receiver_array(receiver_obj, pt0 = starting_coordinates)

#%% plot the scene and save to the the measurement folder
meas_obj.plot_scene(L_x = 0.6, L_y = 0.6, sample_thickness = 0.03, baffle_size = 1.2)

#%% Set the motors for moving
meas_obj.set_motors()

#%% Perform sequential measurement - measured responses will be saved authomatically at your "measured_signals" folder
### Choose repetitions > 1 if you want to average the Impulse responses.
yt_list = meas_obj.sequential_measurement(repetitions = 1)