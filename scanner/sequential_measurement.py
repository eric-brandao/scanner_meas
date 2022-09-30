# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:58:43 2022

Module to control the scanner during sequential impulse response measurements

@author: ericb
"""

# general imports
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as io
from scipy.signal import windows

# Pytta imports
import pytta
from pytta.generate import sweep
from pytta.classes import SignalObj, FRFMeasure
from pytta import ImpulsiveResponse, save, merge

# Arduino imports
from telemetrix import telemetrix

# Receiver class
from receivers import Receiver

# NI imports
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode, RegenerationMode
from nidaqmx.constants import SoundPressureUnits, VoltageUnits
#from MiniRev import MiniRev as mr

#pathh = 'D:/dropbox/Dropbox/2022/meas_29_06/'
# cwd = os.path.dirname(__file__) # Pega a pasta de trabalho atual
# os.chdir(cwd)
#import SSRfunctions as SSR

class ScannerMeasurement():
    """ class to control the scanner
    
    Class used to instantiate objects used to control the sequential
    measurement of impulse responses with UFSM scanner. Arduino is used 
    to control the stepped motors and NI is used for signal acquisition.
    """
    
    def __init__(self, ):
        """

        Parameters
        ----------
        
        """
        self.fs = 51200
        
    def set_arduino_parameters(self, x_pwm = 2, x_dig = 24,
                                y_pwm = 3, y_dig = 26,
                                z_pwm = 4, z_dig = 28,
                                dht = 40):
        """ set arduino parameters
        
        Parameters
        ----------
        x_pwm : int
            PWM pin number for motor x
        x_dig : int
            digital pin number for motor x
        y_pwm : int
            PWM pin number for motor y
        y_dig : int
            digital pin number for motor y
        z_pwm : int
            PWM pin number for motor z
        z_dig : int
            digital pin number for motor z
        """
        self.arduino_params = {'step_pins': [[x_pwm, x_dig],  # x axis
                                [y_pwm, y_dig],  # y axis
                                [z_pwm, z_dig]], # z axis
                  'dht_pin': 40 }
        
        self.exit_flag = 0
        self.u_t = []
        # humidity in air - current and list filled at each meas
        self.humidity_list = []
        self.humidity_current = None
        # temperature - current and list filled at each meas
        self.temperature_current = None
        self.temperature_list = []        
        # ARDUINO BOARD COMMUNICATION
        self.board = telemetrix.Telemetrix()
        
    def set_motors(self,):
        """ Set the motors
        """
        # Motors:
        self.motor_x = self.board.set_pin_mode_stepper(interface=1, 
            pin1 = self.arduino_params['step_pins'][0][0], 
            pin2 = self.arduino_params['step_pins'][0][1], 
            enable = False)
        self.motor_y = self.board.set_pin_mode_stepper(interface=1, 
            pin1 = self.arduino_params['step_pins'][1][0], 
            pin2 = self.arduino_params['step_pins'][1][1], enable=False)
        self.motor_z = self.board.set_pin_mode_stepper(interface=1, 
            pin1 = self.arduino_params['step_pins'][2][0], 
            pin2 = self.arduino_params['step_pins'][2][1], enable=False)

    def set_dht_sensor(self,):
        """ Set sensor to measure humifdity and temperature
        """
        self.board.set_pin_mode_dht(pin = self.arduino_params['dht_pin'], 
            callback = self.the_callback_dht2, dht_type=11)

    def the_callback_dht(self, data):
        """Callback function to measure the current humidity and temperature
        """
        print(data[1])
        if data[1]:
            print(f'DHT Error Report: Pin: {data[2]}, CHECK CONNECTION!')
        else:
            self.humidity_current = data[4]
            self.temperature_current = data[5]
    
    def the_callback_dht2(self, data):
        #global u_t
        
        if self.u_t != []:
            self.u_t = []
        else:
            pass
        if data[1]:
            print(f'dht error report: pin: {data[2]}, check connection!')
        else:
            self.u_t.append(data[4])
            self.u_t.append(data[5])
