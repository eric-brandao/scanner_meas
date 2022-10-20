# -*- coding: utf-8 -*-
"""
MEASUREMENTS - SEPTEMBER  27th, 2022

"""

import sys

import pytta
import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from receivers import Receiver
import time
from pytta.generate import sweep
from nidaqmx.constants import AcquisitionType
from telemetrix import telemetrix
#from MiniRev import MiniRev as mr
from pytta.classes import SignalObj
from pytta import ImpulsiveResponse, save, merge
#pathh = 'D:/dropbox/Dropbox/2022/meas_29_06/'
cwd = os.path.dirname(__file__) # Pega a pasta de trabalho atual
os.chdir(cwd)
import SSRfunctions as SSR

"INSERT AUDIO/MICROPHONE SETTINGS ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- "
fs=51200 # [Hz]
mic_sens = 51.4 # [mV/Pa]
# Excitation signal:
Sweep = sweep(freqMin=1,
          freqMax=int(fs/2),
          samplingRate=fs,
          fftDegree=19,
          startMargin=0.0,
          stopMargin=5.0,
          method='logarithmic',
          windowing='hann')
# Playback/record setup:
NIargs = {'modIn':        {'terminal': 'cDAQ1Mod1/ai0', 
                            'mic_sens': mic_sens, 
                            'currentExc_sensor': 0.0022}, 
          'modOut':       {'terminal': 'cDAQ1Mod3/ao0', 
                            'max_min_val': [-10,10]},
          'timingConfig': {'sampleRate': fs, 
                            'signalExc': Sweep.timeSignal[:,0],
                            'samplesInOut': len(Sweep.timeSignal[:,0]), 
                            'acqsType': AcquisitionType.FINITE},
          'triggerConfig': {'Usage': True, 
                            'terminalTrigger': '/cDAQ1/ai/StartTrigger'},
          }


"ARDUINO SETTINGS ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- "
u_t = []
exit_flag = 0
# Setting up motor and dht pins:
ArduinoParams = {'step_pins': [[2,24],  # eixo X
                               [3,26],  # eixo Y
                               [4,28]], # eixo Z
                 'dht_pin': 40 }

#%% Callback and tasks ender functions

def check_for_tasks():
    if 'input_task' in locals():
        input_task.close()
        del(input_task)
    else:
        pass
    if 'output_task' in locals():
        output_task.close()
        del(output_task)
    else:
        pass

def the_callback_dht(data):
    global u_t
    
    if u_t != []:
        u_t = []
    else:
        pass
    if data[1]:
        # error message
        #date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[4]))
        print(f'DHT Error Report: Pin: {data[2]}, CHECK CONNECTION!')
    else:
       # params = []
        #date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[6]))
        # print(f'DHT Valid Data Report:'
        #       f'Pin: {data[2]} DHT Type: {data[3]} Humidity: {data[4]} Temperature:'
        #       f' {data[5]} Time: {date}')
        u_t.append(data[4]); u_t.append(data[5])

"Stepper callbacks and functions:"

def the_callback(data):
    global exit_flag
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[2]))
    print(f'Motor {data[1]} absolute motion completed at: {date}.')
    exit_flag += 1


def running_callback(data):
    if data[1]:
        print('The motor is running.')
    else:
        print('The motor IS NOT running.')
        
def stepperRun(the_board, motor, dist, microSteps=1600):
    global exit_flag
    if exit_flag > 0:
        exit_flag=0
    else:
        pass
    steps_to_send = int(dist*microSteps/0.008)
    if abs(dist)<=0.16:
        the_board.stepper_set_current_position(0, 0)
        the_board.stepper_set_max_speed(motor, 400)
        the_board.stepper_set_acceleration(motor, 50)
        the_board.stepper_move(motor, steps_to_send)
        the_board.stepper_run(motor, completion_callback=the_callback)
        time.sleep(.2)
        the_board.stepper_is_running(motor, callback=running_callback)
        time.sleep(.2)
        while exit_flag == 0:
            time.sleep(.2)
    else:
        the_board.stepper_set_current_position(0, 0)
        the_board.stepper_set_max_speed(motor, 400)
        the_board.stepper_set_acceleration(motor, 50)
        the_board.stepper_move(motor, int(steps_to_send/2))
        the_board.stepper_run(motor, completion_callback=the_callback)
        time.sleep(.2)
        the_board.stepper_is_running(motor, callback=running_callback)
        time.sleep(.2)
        while exit_flag == 0:
            time.sleep(.2)
        the_board.stepper_set_current_position(0, 0)
        the_board.stepper_set_max_speed(motor, 400)
        the_board.stepper_set_acceleration(motor, 50)
        the_board.stepper_move(motor, int(steps_to_send/2))
        exit_flag=0
        the_board.stepper_run(motor, completion_callback=the_callback)
        time.sleep(.2)
        the_board.stepper_is_running(motor, callback=running_callback)
        time.sleep(.2)
        while exit_flag == 0:
            time.sleep(.2)  

#%% ARDUINO BOARD COMMUNICATION
board = telemetrix.Telemetrix()
# DHT sensor:
board.set_pin_mode_dht(pin=ArduinoParams['dht_pin'], callback=the_callback_dht, dht_type=11)
# Motors:
motorX = board.set_pin_mode_stepper(interface=1, pin1=ArduinoParams['step_pins'][0][0], 
                                    pin2=ArduinoParams['step_pins'][0][1], enable=False)
motorY = board.set_pin_mode_stepper(interface=1, pin1=ArduinoParams['step_pins'][1][0], 
                                    pin2=ArduinoParams['step_pins'][1][1], enable=False)
motorZ = board.set_pin_mode_stepper(interface=1, pin1=ArduinoParams['step_pins'][2][0], 
                                    pin2=ArduinoParams['step_pins'][2][1], enable=False)


#%% SIMPLE MEASUREMENT 

if 'input_task' in locals():
    input_task.close()
    del(input_task)
else:
    pass
if 'output_task' in locals():
    output_task.close()
    del(output_task)
else:
    pass
    
rep=2

"If you want to change the signal excitation --------------------------------------------------------------- "
"        -->> If you do, also you need to change 'Sweep' to 'sweepIR', also, 'NIargs' to 'NIargsIR' -------- "
Sweep = sweep(freqMin=100,
          freqMax=int(fs/2),
          samplingRate=fs,
          fftDegree=19,
          startMargin=0.0,
          stopMargin=0.5,
          method='logarithmic',
          windowing='hann')


NIargs = {'modIn':        {'terminal': 'cDAQ1Mod1/ai0', 
                            'mic_sens': mic_sens, 
                            'currentExc_sensor': 0.0022}, 
          'modOut':       {'terminal': 'cDAQ1Mod3/ao0', 
                            'max_min_val': [-10,10]},
          'timingConfig': {'sampleRate': fs, 
                            'signalExc': Sweep.timeSignal[:,0],
                            'samplesInOut': len(Sweep.timeSignal[:,0]), 
                            'acqsType': AcquisitionType.FINITE},
          'triggerConfig': {'Usage': True, 
                            'terminalTrigger': '/cDAQ1/ai/StartTrigger'},
          }

# Creating and appending the tasks:
input_task, output_task = SSR.get_NI(NIargs, usage='PlayRec')
NIdev = []
NIdev.append(input_task);   NIdev.append(output_task)

# # Play/Rec:
ptIR = SSR.NIpt_PlayRec(NIdev, NIargs, rep)

# # Processing all Impulsive Responses:
# ht=[]
# for i in range(rep):  
#     Ht = ImpulsiveResponse(excitation=Sweep, recording=ptIR[i], samplingRate=fs, regularization=False)
#     Ht.plot_time(xLim=[0.00, 0.016])
#     # Ht.plot_freq()
#     ht.append(Ht)
    
#%% ARRAY - SETUP OF POINTS
receivers = Receiver()
receivers.double_planar_array(x_len=0.4,n_x=2,y_len=0.4,n_y=2, zr=0.015, dz=0.03)
pt0 = np.array([0, 0, 0.039]); "--> Coordinates where the michophone is"
# Changing order of the points:
order1 = SSR.OrderClosest(pt0, receivers.coord)
# Changing signal of X-axis for the scanner move the mic. in the right direction
order1[:,0] = order1[:,0]*-1
#Plot pts:
SSR.plot_scene(order1, sample_size = 0.625, vsam_size=1)
# Creating the matrix with all distances between all points:
standArray = SSR.matrix_stepper(pt0, order1)

standArray[:,1] = standArray[:,1]*-1; #

#np.save(pathh + 'ordened_coord_2832array.npy', order1)
    
#%% ARRAY - MEASUREMENT SETUP
#check_for_tasks()
if 'input_task' in locals():
    input_task.close()
    del(input_task)
else:
    pass
if 'output_task' in locals():
    output_task.close()
    del(output_task)
else:
    pass

# Excitation signal:

# Creating tasks:
input_task, output_task = SSR.get_NI(NIargs, usage='PlayRec')
NIdev = []
NIdev.append(input_task);   NIdev.append(output_task)

#%% ARRAY - MEASUREMENT !!!!!!!!
rep = 2
yt = []
all_u_t = []

for i in range(order1.shape[0]):
    print(f'\n Meas number {i}')
    all_u_t.append(u_t)
    if standArray[i,0] != 0:
        stepperRun(board, motorX, dist=standArray[i,0], microSteps=1600)
        time.sleep(5)
    else:
        pass
    if standArray[i,1] != 0:
        stepperRun(board, motorY, dist=standArray[i,1], microSteps=1600)
        time.sleep(8)
    else:
        pass
    if standArray[i,2] != 0:
        stepperRun(board, motorZ, dist=standArray[i,2], microSteps=1600)
        time.sleep(7)
    else:
        pass
    # pt = SSR.NIpt_PlayRecRAW(NIdev, NIargs, rep)
    # yt.append(pt); del(pt)

print('\n Measurement ended !!! \n')
