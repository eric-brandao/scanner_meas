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

meas_obj = ScannerMeasurement()
meas_obj.set_arduino_parameters(x_pwm = 2, x_dig = 24,
                                y_pwm = 3, y_dig = 26,
                                z_pwm = 4, z_dig = 28,
                                dht = 40)
#%%
meas_obj.set_motors()
#%%
#meas_obj.set_dht_sensor()

#%%
receiver_obj = Receiver()
receiver_obj.double_planar_array(x_len=0.1,n_x=2,y_len=0.1,n_y=2, zr=0.015, 
                                  dz=0.03)

# receiver_obj.random_3d_array(x_len = 1, y_len = 1.0, z_len = 1.0, zr = 0.1, n_total = 10, seed = 0)
# np.random.shuffle(receiver_obj.coord)


pt0 = np.array([0, 0, 0.065]); "--> Coordinates where the michophone is"

meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)

#%%
meas_obj.move_motor(motor_to_move = 'z', dist = -0.005) 

#%%
meas_obj.sequential_movement()

#%%
# meas_obj.stepper_run(meas_obj.motor_x, dist = -0.005)



