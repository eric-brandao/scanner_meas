# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:14:27 2022

@author: ericb
"""
import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')

from sequential_measurement import ScannerMeasurement

#%%

meas_obj = ScannerMeasurement()
meas_obj.set_arduino_parameters(x_pwm = 2, x_dig = 24,
                                y_pwm = 3, y_dig = 26,
                                z_pwm = 4, z_dig = 28,
                                dht = 40)
#%%
meas_obj.set_motors()
#%%
meas_obj.set_dht_sensor()
