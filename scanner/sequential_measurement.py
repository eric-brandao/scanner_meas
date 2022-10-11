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

# utils
import utils

#pathh = 'D:/dropbox/Dropbox/2022/meas_29_06/'
# cwd = os.path.dirname(__file__) # Pega a pasta de trabalho atual
# os.chdir(cwd)
#import SSRfunctions as SSR

class ScannerMeasurement():
    """ class to control the scanner
    
    Class used to instantiate objects used to control the sequential
    measurement of impulse responses with UFSM scanner. Arduino is used 
    to control the stepped motors and NI is used for signal acquisition.
    
    Parameters
    ----------
    fs : int
        sampling rate of the measurement signals
    arduino_params : dict
        dictionary specifying the PWM and digital pins of connection in Arduino
    exit_flag : int
        flag for measurement
    u_t : list
        list containing humidity and temperature of current measurement
    humidity_list : list
        list of all measured humidity
    humidity_current : float
        current value of humidity
    temperature_list : list
        list of all measured temperature        
    temperature_current : float
        current value of temperature
    board : object
        Telemetrix object with Arduino methods and attributes
    motor_x : int
        x-axis motor ID
    motor_y : int
        y-axis motor ID
    motor_z : int
        z-axis motor ID
    receivers : object
        Receiver() object with the ordered receiver coordinates
    stand_array : numpy ndArray
        the x, y, and z distances between the points
    """
    
    def __init__(self,):
        """

        Parameters
        ----------
        
        """
        self.fs = 51200
        self.micro_steps = 1600
        
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
        
    def set_motors(self, pausing_time_array = [5, 8, 7]):
        """ Set the motors
        
        Parameters
        ----------
        pausing_time_array : numpy 1dArray
            1x3 array with the x, y, z pausing times to stablize movement
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
        
        # Motor dictionaries
        self.motor_dict = {'x' : self.motor_x, 'y' : self.motor_y,
                           'z' : self.motor_z}
        self.motor_pause_dict = {'x' : pausing_time_array[0], 
             'y' : pausing_time_array[1], 'z' : pausing_time_array[2]}

    def set_dht_sensor(self,):
        """ Set sensor to measure humifdity and temperature
        """
        self.board.set_pin_mode_dht(pin = self.arduino_params['dht_pin'], 
            callback = self.the_callback_dht, dht_type=11)

    def the_callback_dht(self, data):
        """Callback function to measure the current humidity and temperature
        
        It keeps being called every some seconds. We need a better way to
        call it on demand.
        """
        # print(data[1])
        if data[1]:
            print(f'DHT Error Report: Pin: {data[2]}, CHECK CONNECTION!')
        else:
            self.humidity_current = data[4]
            self.temperature_current = data[5]
    
    def the_callback_dht2(self, data):
        """Callback function to measure the current humidity and temperature
        
        It keeps being called every some seconds. We need a better way to
        call it on demand. Original function.
        """
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
            
    def set_receiver_array(self, receiver_obj, pt0 = None):
        """ set the array of receivers
        
        Parameters
        ----------
        receiver_obj : object Receiver()
            object from Receiver class
        pt0 : numpy 1dArray
            coordinate of the initial pointo of the microphone
        """
        if pt0 is None:
            self.pt0 = np.array([0, 0, 0])
        else:
            self.pt0 = pt0
            
        # original array
        self.receivers = receiver_obj
        # Changing order of the points:
        order1 = utils.order_closest(pt0, self.receivers.coord)
        # new array
        self.receivers.coord = order1
        # creating the matrix with all distances between all points
        self.stand_array = utils.matrix_stepper(self.pt0, self.receivers.coord)
        
    def plot_scene(self, title = '', sample_size = 0.65, vsam_size = 2):
        """ Plot of the scene using matplotlib - not redered
    
        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        """
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        vertices = np.array([[-sample_size/2, -sample_size/2, 0.0],
                             [sample_size/2, -sample_size/2, 0.0],
                             [sample_size/2, sample_size/2, 0.0],
                             [-sample_size/2, sample_size/2, 0.0]])
        verts = [list(zip(vertices[:, 0],
                          vertices[:, 1], vertices[:, 2]))]
        # patch plot
        collection = Poly3DCollection(verts,
                                      linewidths=2, alpha=0.3, edgecolor='black', zorder=2)
        collection.set_facecolor('red')
        ax.add_collection3d(collection)
        
        # plot receiver
        for r_coord in range(self.receivers.coord.shape[0]):
            ax.plot([self.receivers.coord[r_coord, 0]], 
                    [self.receivers.coord[r_coord, 1]], 
                    [self.receivers.coord[r_coord, 2]], 
                    marker='${}$'.format(r_coord),
                    markersize=12, color='black')
        ax.set_title(title)
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, vsam_size))
        ax.view_init(elev=51, azim=-31)

    def stepper_run_base(self, motor, steps_to_send):
        """ Base method to move a motor
        
        Parameters
        ----------
        motor : int
            The motor to be moved
        steps_to_send : int
            The number of steps to move the motor
        """ 
        self.board.stepper_set_current_position(0, 0)
        self.board.stepper_set_max_speed(motor, 400)
        self.board.stepper_set_acceleration(motor, 50)
        self.board.stepper_move(motor, steps_to_send)
        self.board.stepper_run(motor, completion_callback = self.completion_callback)
        self.pause(pausing_time = 0.2)
        self.board.stepper_is_running(motor, callback = self.running_callback)
        self.pause(pausing_time = 0.2)
        while self.exit_flag == 0:
            self.pause(pausing_time = 0.2)
         
    def stepper_run(self, motor, dist = 0.01):
        """ Move the motor
        
        If the distance is larger than 16 [cm], then the movement is executed
        in two steps to avoid bad behaviour
        
        Parameters
        ----------
        motor : int
            The motor to be moved
        dist : float
            Distance in [m] to move the motor
        """
        pre_steps_to_send = dist * self.micro_steps / 0.008
        if abs(dist) <= 0.16:
            steps_to_send = int(pre_steps_to_send)
            self.exit_flag = 0
            self.stepper_run_base(motor, steps_to_send)
        else:
            steps_to_send = int(pre_steps_to_send/2)
            self.exit_flag = 0
            self.stepper_run_base(motor, steps_to_send)            
            self.exit_flag = 0
            self.stepper_run_base(motor, steps_to_send)
    
    def running_callback(self, data):
        """Callback function to inform if the motor is moving or not
        """
        if data[1]:
            print('The motor is running.')
        else:
            print('The motor IS NOT running.')            
    
    def completion_callback(self, data):
        """Callback function to inform that the movement is complete
        
        Informs that the movement of a certain motor is complete and changes
        the value of the exit_flag variable in the class
        """
        # global exit_flag
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[2]))
        print(f'Motor {data[1]} absolute motion completed at: {date}.')
        self.exit_flag += 1
        
    def pause(self, pausing_time = 0.2):
        """ Pause for a few seconds
        
        Used to stabilize the measurement system.
        
        Parameters
        ----------
        time : float
            time to pause
        """
        time.sleep(pausing_time)
    
    def move_motor(self, motor_to_move = 'x', dist = 0.01):
        """ Move motor x
        
        Parameters
        ----------
        motor : float
            can be either 'x', 'y' or 'z' - specifying the motor to be moved
        dist : float
            distance along x-axis in [m]
        micro_steps : int
            number of micro steps
        """
        self.stepper_run(self.motor_dict[motor_to_move], dist = dist)
        self.pause(pausing_time = self.motor_pause_dict[motor_to_move])
        
    def sequential_movement(self,):
        """ Move all motors sequentially through the array positions
        """
        for i in range(self.receivers.coord.shape[0]):
            print(f'\n Meas number {i+1} of {self.receivers.coord.shape[0]}')
            # all_u_t.append(u_t)
            if self.stand_array[i, 0] != 0:
                self.move_motor(motor_to_move = 'x',
                                dist = self.stand_array[i, 0])
                # stepperRun(board, motorX, dist=standArray[i,0], microSteps=1600)
                # time.sleep(5)
            # else:
            #     pass
            if self.stand_array[i, 1] != 0:
                self.move_motor(motor_to_move = 'y',
                                dist = self.stand_array[i, 1])
                # stepperRun(board, motorY, dist=standArray[i,1], microSteps=1600)
                # time.sleep(8)
            # else:
            #     pass
            if self.stand_array[i, 2] != 0:
                self.move_motor(motor_to_move = 'z',
                                dist = self.stand_array[i, 2])
                # stepperRun(board, motorZ, dist=standArray[i,2], microSteps=1600)
                # time.sleep(7)
            # else:
            #     pass

        print('\n Measurement ended !!! \n')
        


    
