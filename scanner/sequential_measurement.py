# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:58:43 2022

Module to control the scanner during sequential impulse response measurements

@author: ericb
"""


# general imports
import sys
import os
from pathlib import Path
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as io
from scipy.signal import windows, resample, chirp
from datetime import datetime

# Pytta imports
import pytta
# from pytta.generate import sweep
# from pytta.classes import SignalObj, FRFMeasure
# from pytta import ImpulsiveResponse, save, merge
from ni_measurement import NIMeasurement

# Arduino imports
from telemetrix import telemetrix

# Receiver class
from receivers import Receiver
from sources import Source

# NI imports
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode, RegenerationMode, Coupling
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
    
    def __init__(self, main_folder = 'D:', name = 'samplename',                 
            fs = 51200, fft_degree = 18, 
            start_stop_margin = [0.1, 1.0], mic_sens = None,
            x_pwm_pin = 2, x_digital_pin = 24,
            y_pwm_pin = 3, y_digital_pin = 26,
            z_pwm_pin = 5, z_digital_pin = 28,
            dht_pin = 40, pausing_time_array = [8, 8, 8],
            material = None, material_type = 'melamine_L60cm_d4cm',
            temperature = 20, humidity = 0.5,
            microphone_type = 'Behringer ECM 8000',
            audio_interface = 'M-audio Fast Track Pro',
            amplifier = 'BK 2718',
            source_type = 'spherical speaker', source = None, 
            start_new_measurement = True,
            sound_card_measurement = True):
        """

        Parameters
        ----------
        main_folder : str
            main folder for the measurements
        name: str
            name of the measurement
        fs : int
            sampling rate
        fft_degree : int
            degree of measured signals - N of samples is 2**fft_degree
        start_stop_margin : list
            list with start margin and stop margin
        mic_sens : float
            microphone sensitivity [mV/Pa]
        x_pwm_pin : int
            pin number of PWM on x-axis
        x_digital_pin : int
            pin number of digital signal on x-axis
        y_pwm_pin : int
            pin number of PWM on y-axis
        y_digital_pin : int
            pin number of digital signal on y-axis
        z_pwm_pin : int
            pin number of PWM on z-axis
        z_digital_pin : int
            pin number of digital signal on z-axis
        dht_pin : int
            pin number of dht reading
        pausing_time_array : numpy 1dArray
            1x3 array with the x, y, z pausing times to stablize movement
        material : object
            Object containing pre-measured material properties (e.g. flow resistivity)
        material_type : str 
            String describing the test material - include relevant info (type, size, thickness)
        temperature : float
            Measured temperature in ºC (later we can measure it with a sensor at each coordinate) 
        humidity : float
            Measured humidity [-] lin scale (later we can measure it with a sensor at each coordinate) 
        microphone_type : str
            String describing microphone model
        audio_interface : str
            String describing the audio interface (DAQ) model
        amplifier : str
            String describing the power amplifier model
        source_type : str
            String describing the source type (e.g. spherical speaker, dipole, ...)
        source : obj
            source object containing its coordinates
        start_new_measurement : bool
            Boolean variable to inform if you want to start a new measurement or not. If True, it gives
            permission to the code to create a new "measured_signals" folder.
        """
        # folder checking
        self.main_folder = Path(main_folder)
        self.name = name
        self.check_main_folder(generate_folder = start_new_measurement)
        
        # audio signals checking
        self.fs = fs
        self.fft_degree = fft_degree
        self.start_margin = start_stop_margin[0]
        self.stop_margin = start_stop_margin[1]
        self.micro_steps = 1600
        self.mic_sens = mic_sens
        
        # arduino main parameters
        self.arduino_params = {'step_pins': [[x_pwm_pin, x_digital_pin],  # x axis
                                [y_pwm_pin, y_digital_pin],  # y axis
                                [z_pwm_pin, z_digital_pin]], # z axis
                               'dht_pin': dht_pin}
        
        self.pausing_time_array = pausing_time_array
        
        self.exit_flag = 0
        self.u_t = []
        # humidity in air - current and list filled at each meas
        self.humidity_list = []
        self.humidity_current = None
        # temperature - current and list filled at each meas
        self.temperature_current = None
        self.temperature_list = []
        self.temperature = temperature
        self.humidity = humidity
        
        # Felipe e João added
        self.initError = 0
        
        # material
        self.material = material
        self.material_type = material_type
        
        # Instrumentation
        self.microphone_type = microphone_type
        self.audio_interface = audio_interface
        self.sound_card_measurement = sound_card_measurement
        self.amplifier = amplifier
        self.source_type = source_type
        self.source = source
        
        # saving the control object        
        # self.save() # save initialization
    
    def set_measurement_date(self,):
        """ Use datetime to set the measurement date
        """
        self.date_of_measurement = datetime.today()
        self.save()
        
    def get_measurement_date(self,):
        """ Use datetime to get the dd/mm/yyyy of measurement
        """
        print("Date (dd/mm/yyyy) of measurement is {}/{}/{}".format(self.date_of_measurement.day,
                                                       self.date_of_measurement.month,
                                                       self.date_of_measurement.year))
        
    def check_main_folder(self, generate_folder = True):
        """ Check if the main folder already exists or create new folder
        """
        folder_to_test = self.main_folder / self.name
        if folder_to_test.exists() and generate_folder:
            print('Measurement path already exists. Proceed with care as you may loose data. Use this object to read only.')
        elif folder_to_test.exists() and generate_folder == False:
            print("You probably want to load measurement files that you copied or moved. I'll not create any new folders. Use this object to read only.")
        elif folder_to_test.exists() == False and generate_folder:
            folder_to_test.mkdir(parents = False, exist_ok = False)
            # response signals
            measured_signals_folder = folder_to_test / 'measured_signals'
            measured_signals_folder.mkdir(parents = False, exist_ok = False)
            # impulse responses
            impulse_responses_folder = folder_to_test / 'impulse_responses'
            impulse_responses_folder.mkdir(parents = False, exist_ok = False)
        else:
            print("The measurement files you seek do not exist in the folder you specified")

    # def ni_set_config_dicts(self, input_dict = dict(terminal = 'cDAQ1Mod1/ai0', 
    #          mic_sens = 51.4, current_exc_sensor = 0.0022,
    #          max_min_val = [-5, 5]),
    #          output_dict = dict(terminal = 'cDAQ1Mod3/ao0', 
    #          max_min_val =  [-10,10])):
    #     """ sets NI configuration dictionaries
    #     """
    #     self.input_dict = input_dict
    #     self.output_dict = output_dict     

    def pytta_list_devices(self,):
        """ Liste pytta devices for choice
        """
        print(pytta.list_devices())
        
    def pytta_set_device(self, device):
        """Choose your input and output devices for a full pytta measurement
        
        Parameters
        ----------
        device : int
            integer specifying your measurement device (from pytta.list_devices())
        """
        self.device = device
        
    
    def set_meas_sweep(self, method = 'logarithmic', freq_min = 1,
                       freq_max = None, n_zeros_pad = 200, save_xt = True):
        """Set the input signal object
        
        The input signal is called "xt". This is to ease further
        implementation. For example, if you want to set a random signal,
        you can also call it "xt", and pass it to the same method that 
        computes the IR
        
        Parameters
        ----------
        method : str
            method of sweep
        freq_min : float
            minimum frequency of the sweep
        freq_max : float or None
            maximum frequency of the sweep. Default is None, which sets
            it to fs/2
        """
        # sweep metadata
        self.freq_min = freq_min
        if freq_max is None:
            self.freq_max = int(self.fs/2)
        else:
            self.freq_max = freq_max
        
        self.method = method
        self.n_zeros_pad = n_zeros_pad
        
        # set pytta sweep with zero padding if wanted
        xt = pytta.generate.sweep(freqMin = self.freq_min,
          freqMax = self.freq_max, samplingRate = self.fs,
          fftDegree = self.fft_degree, startMargin = self.start_margin,
          stopMargin = self.stop_margin,
          method = self.method, windowing='hann')
        new_xt = np.zeros(len(xt.timeSignal[:,0]) + n_zeros_pad)
        new_xt[:len(xt.timeSignal[:,0])] = xt.timeSignal[:,0]

        self.xt = pytta.classes.SignalObj(
            signalArray = new_xt, 
            domain='time', samplingRate = self.fs, freqMin = self.freq_min,
              freqMax = self.freq_max)

        self.Nsamples = len(self.xt.timeSignal[:,0])

        # save the sweep
        #print(self.main_folder)
        if save_xt:
            complete_path = self.main_folder / self.name / 'measured_signals'
            pytta.save(str(complete_path / 'xt.hdf5'), self.xt)
    
    def ni_initializer(self, buffer_size = 2**8):
        """ Initialize NI for measurement
        """
        self.buffer_size = buffer_size
        self.ni_control_obj = NIMeasurement(reference_sweep = self.xt, 
                                            fs = self.xt.samplingRate, buffer_size = buffer_size)
        self.ni_control_obj.get_system_and_channels()

    
    def ni_set_output_channels(self, out_channel_to_ni = 3, out_channel_to_amp = 1, ao_range = 10.0):
        """ Set NI output channels
        """
        self.out_channel_to_ni = out_channel_to_ni
        self.out_channel_to_amp = out_channel_to_amp
        self.ao_range = ao_range
        self.ni_control_obj.set_output_channels(out_channel_to_ni = self.out_channel_to_ni, 
                                    out_channel_to_amp = self.out_channel_to_amp, 
                                    ao_range = self.ao_range)
        

    def ni_set_input_channels(self, in_channel_ref_onrec = 3, in_channel_sensor_onrec = [0],
                               ai_range = 1, sensor_sens = 50, sensor_current = 2.2e-3):
        """ Set NI output channels
        """
        self.in_channel_sensor = 2
        self.in_channel_ref = 1
        self.in_channel_ref_onrec = in_channel_ref_onrec
        self.in_channel_sensor_onrec = in_channel_sensor_onrec
        self.ai_range = ai_range
        self.sensor_sens = sensor_sens
        self.sensor_current = sensor_current
        self.ni_control_obj.set_input_channels(in_channel_ref = self.in_channel_ref_onrec, 
                                               in_channel_sensor = self.in_channel_sensor_onrec,
                                               ai_range = self.ai_range, 
                                               sensor_sens = self.sensor_sens, 
                                               sensor_current = self.sensor_current)
        # self.save()
        # self.load()
        
        
    # def ni_set_play_rec_tasks(self, ):
        
    #     self.rn = np.random.randint(0, high = 1000)        
    #     # if hasattr(self, 'input_task'):
    #     #     self.input_task.close()
    #     #     del(self.input_task)
        
    #     self.ni_get_input_task(input_type = 'mic')
        
    #     # if hasattr(self, 'output_task'):
    #     #     self.output_task.close()
    #     #     del(self.output_task)
            
    #     self.ni_get_output_task()
        
    
    # def ni_get_input_task(self, input_type = 'mic'):
    #     """ Get input task for NI
        
    #     Parameters
    #     ----------
    #     input_type : str
    #         string with type of recording. Can be either 'mic' for microphone
    #         or 'voltage' to configure by pass measurement. Default or any str sets
    #         to 'mic'
    #     """
    #     #input_unit = SoundPressureUnits.PA
    #     # Max of sound card for dBFS
    #     Coupling.AC
    #     self.max_val_dbFS = self.input_dict['max_min_val'][1]
    #     # Instantiate NI object
    #     self.input_task = nidaqmx.Task(new_task_name = 'intask' + str(self.rn))
    #     # Configure input signal
    #     if input_type == 'mic':
    #         self.input_task.ai_channels.add_ai_microphone_chan(
    #             self.input_dict['terminal'], 
    #             units = SoundPressureUnits.PA, 
    #             mic_sensitivity = self.input_dict['mic_sens'],
    #             current_excit_val = self.input_dict['current_exc_sensor'])
    #     elif input_type == 'voltage':
    #         self.input_task.ai_channels.add_ai_voltage_chan(
    #             self.input_dict['terminal'],
    #             min_val = self.input_dict['max_min_val'][0], 
    #             max_val = self.input_dict['max_min_val'][1],
    #             units = VoltageUnits.VOLTS)
    #     else:
    #         self.input_task.ai_channels.add_ai_microphone_chan(
    #             self.input_dict['terminal'], 
    #             units = SoundPressureUnits.PA, 
    #             mic_sensitivity = self.input_dict['mic_sens'],
    #             current_excit_val = self.input_dict['current_exc_sensor'])
            
    #     self.input_task.timing.cfg_samp_clk_timing(self.fs,
    #         sample_mode = AcquisitionType.CONTINUOUS, #AcquisitionType.FINITE, 
    #         samps_per_chan = self.Nsamples)#self.Nsamples
            
    #     # return input_task
    
    # def ni_get_output_task(self, ):
    #     """ Get output task for NI
    #     """
    #     Coupling.AC
    #     self.output_task = nidaqmx.Task(new_task_name = 'outtask' + str(self.rn))
        
    #     self.output_task.ao_channels.add_ao_voltage_chan(
    #         self.output_dict['terminal'],
    #         min_val = self.output_dict['max_min_val'][0], 
    #         max_val = self.output_dict['max_min_val'][1],
    #         units = VoltageUnits.VOLTS)
        
    #     self.output_task.timing.cfg_samp_clk_timing(self.fs,
    #         sample_mode = AcquisitionType.CONTINUOUS, #AcquisitionType.FINITE,
    #         samps_per_chan = self.Nsamples)
        
    #     self.output_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION #RegenerationMode.ALLOW_REGENERATION
        
    #     self.output_task.write(self.xt.timeSignal[:,0])
    #     # print(a)
    #     self.output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/cDAQ1/ai/StartTrigger')
        
    #     # return output_task
    
    # def ni_play_rec(self,):
    #     """Measure response signal using NI
        
    #     Returns
    #     ----------
    #     yt_rec_obj : pytta object
    #         output signal
    #     """
    #     self.ni_set_play_rec_tasks()
    #     # Coupling.AC
    #     # Initialize for measurement
    #     print('Acqusition started')
    #     self.output_task.start()
    #     self.input_task.start()
    #     # Measure
    #     master_data = self.input_task.read(
    #         number_of_samples_per_channel = self.Nsamples,
    #         timeout = 2*round(self.Nsamples/self.fs, 2))
    #     # Stop measuring
    #     self.input_task.stop()
    #     self.input_task.close()
    #     self.output_task.stop()
    #     self.output_task.close()
    #     # Get list as array
    #     yt_rec = np.asarray(master_data)
    #     # Print message
    #     dBFS = round(20*np.log10(np.amax(np.abs(yt_rec))/self.max_val_dbFS), 2)
    #     print('Acqusition ended: {} dBFS'.format(dBFS))
    #     # Pass to pytta
    #     yt_rec_obj = pytta.classes.SignalObj(
    #         signalArray = yt_rec, 
    #         domain='time', freqMin = self.freq_min, 
    #         freqMax = self.freq_max, samplingRate = self.fs)
        
    #     return yt_rec_obj
    
    def pytta_play_rec_setup(self, in_channel = [1, 2], out_channel = [1, 2],
                             in_channel_ref = 2, in_channel_sensor = 1,
                             output_amplification = -3):
        """ Configure measurement of response signal using pytta and sound card
        """
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_channel_ref = in_channel_ref
        self.in_channel_sensor = in_channel_sensor
        self.output_amplification = output_amplification
        
        self.pytta_meas = pytta.generate.measurement('playrec',
            excitation = self.xt,
            samplingRate = self.fs,
            freqMin = self.freq_min,
            freqMax = self.freq_max,
            device = self.device,
            inChannels = self.in_channel,
            outChannels = self.out_channel,
            outputAmplification = self.output_amplification)

    def pytta_play_rec(self,):
        """ Measure response signal using pytta and sound card
        
        Returns
        ----------
        yt_rec_obj : pytta object
            output signal
        """
        print('Acqusition started')
        yt_rec_obj = self.pytta_meas.run()
        print('Acqusition ended')
        return yt_rec_obj
    
    def ni_play_rec(self,):
        """ Measure response signal using pytta and NI
        
        Returns
        ----------
        yt_rec_obj : pytta object
            output signal
        """
        yt_rec_obj = self.ni_control_obj.play_rec()
        print('Acqusition ended')
        return yt_rec_obj
    
    # def pytta_measure_loopback(self,):
    #     """ Measure system loopback of your sound card if you want
        
    #     Applicable to audio interfaces - should be flat inside the sweep range
    #     """
    #     print("I hope you are measuring your loopback IR")
    #     yt_lb = self.pytta_play_rec()
    #     ht_lb = self.ir(yt_lb, regularization = True)
    #     ht_lb.IR.plot_freq(xLim = [20, 20000])
    #     ht_lb.plot_time()
    #     # saving loophack IR
    #     complete_path = self.main_folder / self.name / 'impulse_responses'
    #     pytta.save(str(complete_path / 'ht_lb.hdf5'), ht_lb.IR)
    
    def ir(self, yt, regularization = True, deconv_with_rec = True):
        """ Computes the impulse response of a given output
        
        Parameters
        ----------
        yt : pytta object
            output measured signal(s). It can be made solely of the recorded signal,
            if you did not measure the reference output; or it can be a 2 channel measurement
            made of the mic (sensor) channel and the reference channel.
        regularization : bool
            Do you want noise in your h(t)? No, right - so leave it on. The False value is just
            to show students what kinds of catatrophes can happen.
        deconv_with_rec : bool
            If True the reference signal is the recorded sweep. It is important to do that if you 
            want to sync your IR. You cal set it to False. THen your reference signal is the 
            mathematical sweep you designed (no garantee of sync).            
        """
        yt_list = yt.split()
        if deconv_with_rec:
            ht = pytta.ImpulsiveResponse(excitation = yt_list[self.in_channel_ref-1], 
                 recording = yt_list[self.in_channel_sensor-1], samplingRate = self.fs, 
                 regularization = regularization, freq_limits = [self.freq_min, self.freq_max])
        else:
            ht = pytta.ImpulsiveResponse(excitation = self.xt, 
                 recording = yt_list[self.in_channel_sensor-1], 
                 samplingRate = self.fs, regularization = regularization, 
                 freq_limits = [self.freq_min, self.freq_max])
        return ht

        
    def set_arduino_parameters(self,):
        """ set arduino parameters
        
        """        
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
        
    def set_motors(self, ):
        """ Set the ARDUINO board and motors
        
        """
        print('Pre-setting the motors and arduino controller.')
        self.board = telemetrix.Telemetrix()
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
        
        # Temperature and humidity
        # self.set_dht_sensor()
        # Motor dictionaries
        self.motor_dict = {'x' : self.motor_x, 'y' : self.motor_y,
                           'z' : self.motor_z}
        self.motor_pause_dict = {'x' : self.pausing_time_array[0], 
             'y' : self.pausing_time_array[1], 
             'z' : self.pausing_time_array[2]}
        
        # print("We are moving y motor to make sure everything is working.")
        # self.move_motor(self, motor_to_move = 'y', dist = -0.001)
        # self.move_motor(self, motor_to_move = 'y', dist = 0.001)
        
        # Alteração Felipe e João
        print("Setting up motors... Moving y-axis +1mm and -1mm")
        self.move_motor(motor_to_move = 'y', dist = -0.001)
        self.move_motor(motor_to_move = 'y', dist = 0.001)
        print("Setup complete! y-axis moved correctly")

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
        # save and re-load to reconstruct xt
        print("I am saving everything you configured so far. I'll also load the x(t)")
        self.save()
        
        self.load()
        
    def plot_scene(self, L_x = 0.6, L_y = 0.6, sample_thickness = 0.1,
                   baffle_size = 1.2, elev = 30, azim = 45):
        """ Plot of the scene using matplotlib - not redered
    
        Parameters
        ----------
        sample_size : float
            Sample size.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d') #fig.gca()#projection='3d'
        
        list_of_sample_verts = []
        # Sample top
        list_of_sample_verts.append(np.array([[-L_x/2, -L_y/2, 0.0],
                             [L_x/2, -L_y/2, 0.0],
                             [L_x/2, L_y/2, 0.0],
                             [-L_x/2, L_y/2, 0.0]]))
        #lx 1                     
        list_of_sample_verts.append(np.array([[-L_x/2, -L_y/2, 0.0],
                             [L_x/2, -L_y/2, 0.0],
                             [L_x/2, -L_y/2, -sample_thickness],
                             [-L_x/2, -L_y/2, -sample_thickness]]))
        # lx 2
        list_of_sample_verts.append(np.array([[-L_x/2, L_y/2, 0.0],
                              [-L_x/2, L_y/2, -sample_thickness],
                              [L_x/2, L_y/2, -sample_thickness],
                              [L_x/2, L_y/2, 0.0]]))
        #ly 1
        list_of_sample_verts.append(np.array([[-L_x/2, -L_y/2, 0.0],
                              [-L_x/2, -L_y/2, -sample_thickness],
                              [-L_x/2, L_y/2, -sample_thickness],
                              [-L_x/2, L_y/2, 0.0]]))
        #ly 2
        list_of_sample_verts.append(np.array([[L_x/2, -L_y/2, 0.0],
                             [L_x/2, L_y/2, 0.0],
                             [L_x/2, L_y/2, -sample_thickness],
                             [L_x/2, -L_y/2, -sample_thickness]]))
        
        for jv in np.arange(len(list_of_sample_verts)):
            verts = [list(zip(list_of_sample_verts[jv][:, 0],
                          list_of_sample_verts[jv][:, 1], list_of_sample_verts[jv][:, 2]))]
            # patch plot
            collection = Poly3DCollection(verts,
                                          linewidths=0.5, alpha=0.5, edgecolor='tab:blue', zorder=2)
            collection.set_facecolor('tab:blue')
            ax.add_collection3d(collection)
        
        # Baffle
        baffle = np.array([[-baffle_size/2, -baffle_size/2, -sample_thickness],
                         [baffle_size/2, -baffle_size/2, -sample_thickness],
                         [baffle_size/2, baffle_size/2, -sample_thickness],
                         [-baffle_size/2, baffle_size/2, -sample_thickness]])
        
        verts = [list(zip(baffle[:, 0], baffle[:, 1], baffle[:, 2]))]
        # patch plot
        collection = Poly3DCollection(verts,
                                      linewidths=0.5, alpha=0.5, edgecolor='grey', zorder=2)
        collection.set_facecolor('grey')
        ax.add_collection3d(collection)
        
        # plot source
        if self.source != None:
            ax.scatter(self.source.coord[0, 0],self.source.coord[0, 1], self.source.coord[0, 2],
                       s=200, marker = '*', color='red', alpha = 0.5)
        
        
        #plot receiver
        for r_coord in range(self.receivers.coord.shape[0]):
            ax.scatter([self.receivers.coord[r_coord, 0]], 
                    [self.receivers.coord[r_coord, 1]], 
                    [self.receivers.coord[r_coord, 2]], 
                    marker='o', s=12, color='blue', alpha = 0.7)
        
        ax.set_title("Measurement scene")
        ax.set_xlabel(r'$x$ [m]')
        ax.set_xticks([-baffle_size/2, -L_x/2, L_x/2, baffle_size/2])
        ax.set_ylabel(r'$y$ [m]')
        ax.set_yticks([-L_y/2, L_y/2])
        ax.set_zlabel(r'$z$ [m]')
        ax.set_zticks([-sample_thickness, 1.0, 2, 3])
        ax.grid(False)
        ax.set_xlim((-baffle_size/2, baffle_size/2))
        ax.set_ylim((-baffle_size/2, baffle_size/2))
        ax.set_zlim((-sample_thickness, baffle_size))
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        filename = 'measurement_scene.pdf'
        plt.savefig(fname = self.main_folder /self.name / filename, format='pdf', dpi = 300)
        plt.show()

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
        """ Move motor x, y or z
        
        Parameters
        ----------
        motor : float
            can be either 'x', 'y' or 'z' - specifying the motor to be moved
        dist : float
            distance along x-axis in [m]
        micro_steps : int
            number of micro steps
        """
        self.stepper_run(self.motor_dict[motor_to_move], dist = -dist)
        self.pause(pausing_time = self.motor_pause_dict[motor_to_move])
        
    def move_motor_xyz(self, distance_vector):
        """ Move three motors sequentially
        
        Parameters
        ----------
        distance_vector : numpy 1dArray
            vector containing the x, y, z distances to displace each motor        
        """
        keys = list(self.motor_dict.keys())
        for axis in range(3):
            if distance_vector[axis] != 0:
                self.move_motor(motor_to_move = keys[axis],
                                dist = distance_vector[axis])        
        
    def sequential_movement(self,):
        """ Move all motors sequentially through the array positions
        """
        for jrec in range(self.receivers.coord.shape[0]):
            print(f'\n Position number {jrec+1} of {self.receivers.coord.shape[0]}')
            
            self.move_motor_xyz(self.stand_array[jrec,:])
        
        print('\n Moving ended. I will shut down the board instance! \n')
        self.board.shutdown()
        
    def sequential_measurement(self, meas_with_ni = False, 
                               repetitions = 1,
                               plot_frf = False):
        """ Move all motors sequentially through the array positions
        
        Parameters
        ----------
        meas_with_ni : bool
            whether to measure wit NI or not. Default is True
        repetitions : int
            number of repeated measurements
        plot_frf : bool
            whether to plot the FRF or not
        """
        self.repetitions = repetitions
        #yt_list = []
        for jrec in range(self.receivers.coord.shape[0]):
            # Move the motor
            self.move_motor_xyz(self.stand_array[jrec,:])
            # Take temperature and pressure
            ####
            # Take measurement and save it #ToDo
            
            # y_rep_list = []
            for jmeas in range(self.repetitions):
                print('\n Receiver {} of {} (Repeat {} of {})'.format(
                    jrec+1, self.receivers.coord.shape[0], jmeas+1, self.repetitions))
                if meas_with_ni:
                    yt_obj = self.ni_play_rec()
                else: # measure with pytta
                    yt_obj = self.pytta_play_rec()
                
                # if plot_frf:
                #     self.plot_spk(yt_obj)
                
                #y_rep_list.append(yt_obj)
                # ptta saving
                filename = 'rec' + str(int(jrec)) +\
                    '_m' + str(int(jmeas)) + '.hdf5'
                complete_path = self.main_folder / self.name / 'measured_signals'
                pytta.save(str(complete_path / filename), yt_obj)
                # plot FRF
                
            # append all to main list
            #yt_list.append(y_rep_list)
            # Take temperature and humidity
            self.temperature_list.append(self.temperature_current)
            self.humidity_list.append(self.humidity_current)
        # update control object
        self.board.shutdown()
        self.save()        
        print('\n Measurement ended. I will shut down the board instance! \n')
        #return yt_list
    
    
    def take_measurements(self, repetitions = 1, meas_name = 'name'):
        """ Move all motors sequentially through the array positions
        
        Parameters
        ----------
        repetitions : int
            number of repeated measurements
        meas_name : name
            measurement name
        """
        self.repetitions = repetitions
        for jmeas in range(self.repetitions):
            print('Taking measurement #{}'.format(jmeas))
            yt_obj = self.pytta_play_rec()
            # # ptta saving
            # filename = meas_name +\
            #     '_m' + str(int(jmeas)) + '.hdf5'
            # complete_path = self.main_folder / self.name / 'measured_signals'
            # pytta.save(str(complete_path / filename), yt_obj)
            # plot FRF
        return yt_obj
    
    def plot_spk(self, yt_obj):
        """ Plot magnitude of FRF
        """
        # compute FRF
        ht = self.ir(yt = yt_obj, regularization = True)
        freq = ht.irSignal.freqVector
        mag_h = 20*np.log10(np.abs(ht.irSignal.freqSignal))
        fig = plt.figure(num = 1, figsize = (7,5))
        plt.semilogx(freq, mag_h)
        
        
        
    
    def delete_unpickleable_vars(self,):
        """ Delete pytta, NI, and telemetrix objects
        
        These instances are unpickleable
        
        Returns
        ----------
        temp_dict : dict
            Pickleable dictionary        
        """
        temp_dict =  self.__dict__   
        if hasattr(self, 'xt'):
            del temp_dict['xt']
        if hasattr(self, 'pytta_meas'):
            del temp_dict['pytta_meas']
        if hasattr(self, 'board'):
            del temp_dict['board']
        if hasattr(self, 'ni_control_obj'):
            del temp_dict['ni_control_obj']
        
        return temp_dict
    
    def save(self,):
        """Saves the measurement control object as pickle

        """
        pickle_name = self.name + '.pkl'
        path_filename = self.main_folder / self.name / pickle_name
        save_dict = self.delete_unpickleable_vars()
        with path_filename.open(mode = 'wb') as f:
            pickle.dump(save_dict, f, 2)
        f.close()
        
    def load(self,):
        """Loads the measurement control object as pickle

        """
        pickle_name = self.name + '.pkl'
        path_filename = self.main_folder / self.name / pickle_name
        with path_filename.open(mode = 'rb') as f:
            tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

        try:
            complete_path = self.main_folder / self.name / 'measured_signals'
            med_dict = pytta.load(str(complete_path / 'xt.hdf5'))
            keyslist = list(med_dict.keys())
            self.xt = med_dict[keyslist[0]]
        except:
            self.set_meas_sweep(method = self.method, 
                    freq_min = self.freq_min, freq_max = self.freq_max,
                    n_zeros_pad = self.n_zeros_pad, save_xt = False)
        self.Nsamples = len(self.xt.timeSignal[:,0])
        if self.sound_card_measurement:
            self.pytta_play_rec_setup(in_channel = self.in_channel, out_channel = self.out_channel, 
                                      output_amplification = self.output_amplification)
        else:
            print('measured_signals')
            self.ni_initializer(buffer_size = self.buffer_size)
            self.ni_set_output_channels(out_channel_to_ni = self.out_channel_to_ni, 
                                        out_channel_to_amp = self.out_channel_to_amp, 
                                        ao_range = self.ao_range)
            self.ni_set_input_channels(in_channel_ref_onrec = self.in_channel_ref_onrec, 
                                    in_channel_sensor_onrec = self.in_channel_sensor_onrec,
                                    ai_range = self.ai_range, 
                                    sensor_sens = self.sensor_sens, 
                                    sensor_current = self.sensor_current)
        self.__dict__.update(tmp_dict)
        
        
    def load_meas_files(self,):
        """Load all measurement files
        """
        yt_list = []
        for jrec in range(self.receivers.coord.shape[0]):
            y_rep_list = []
            for jmeas in range(self.repetitions):
                filename = 'rec' + str(int(jrec)) +\
                        '_m' + str(int(jmeas)) + '.hdf5'
                complete_path = self.main_folder / self.name / 'measured_signals'
                med_dict = pytta.load(str(complete_path / filename))
                # print(med_dict)
                y_rep_list.append(med_dict['repetitions'])
            yt_list.append(y_rep_list)
            
        return yt_list
        


    
