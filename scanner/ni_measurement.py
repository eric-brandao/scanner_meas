# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:31:35 2024

@author: Admin
"""

# general imports
# import sys
# import pickle
import numpy as np
import matplotlib.pyplot as plt
# from warnings import warn  # , filterwarnings
# from tqdm import tqdm


# import scipy.io as io
# from scipy.signal import windows, resample, chirp
# from datetime import datetime

# Pytta imports
import pytta
# from pytta.generate import sweep
# from pytta.classes import SignalObj, FRFMeasure
# from pytta import ImpulsiveResponse, save, merge

# Arduino imports

# NI imports
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode, RegenerationMode, Coupling
from nidaqmx.constants import VoltageUnits, CurrentUnits, SoundPressureUnits, ForceUnits, AccelUnits,  VelocityUnits
#from MiniRev import MiniRev as mr

# utils
import utils

class NIMeasurement(object):
    """ class to control the NI daq
    
    Parameters
    ----------
    fs : int
        sampling rate of the measurement signals
    """
    
    def __init__(self, fs = 51200, buffer_size = 2**10,
                 fft_degree = 18, time_length = None,
                 reference_signal = None):
        """

        Parameters
        ----------
        fs : int
            sampling rate
        buffer_size : int
            The size of your buffer on playback and/or recording. 
            It should be a power of 2 (since your playback
            signal has size of a power of 2. This means that a integer number of blocks are played-back)
        fft_degree : int
            This is used to specify the number of samples of your playback and or recorded signals.
            The total number of samples is 2**fft_degree
        time_length : float
            This is used to specify the time duration of the playback and or recorded signals in [s].
            It will overide fft_degree
        reference_sweep : Pytta Object
            Pytta Object containing a reference sweep for impulse response measurements. 
            Specifying the sweep overides all input parameters
        """
        # Sampling rate
        if fs is None:
            raise ValueError("You must specify the sampling rate of NI device")
        else:        
            self.fs = fs
            
        # Buffer size
        if buffer_size is None:
            raise ValueError("You must specify a valid buffer size.")
        else:
            self.buffer_size = buffer_size
        # elif np.log2(buffer_size).is_integer():
        #     self.buffer_size = buffer_size
        # else:
        #     self.buffer_size = int(2**np.floor(np.log2(buffer_size)))
            
        # fft_degree and time_length
        if fft_degree is None and time_length  is None and reference_signal is None:
            raise ValueError("You must specify either a fft_degree or time_length of the singal. Or specify the reference signal to play")
        elif time_length is None:        
            self.fft_degree = fft_degree
            self.num_of_samples = 2**self.fft_degree
            self.time_length = (self.num_of_samples-1)/self.fs
        else:
            self.time_length = time_length # maybe this needs small ajustment (start at 0)
            self.num_of_samples = int(self.time_length*self.fs)
            self.fft_degree = np.log2(self.num_of_samples)
                    
        # Reference sweep - overides the rest
        if reference_signal is not None:
            self.xt = reference_signal
            self.fft_degree = reference_signal.fftDegree
            self.time_length = reference_signal.timeLength
            self.fs = reference_signal.samplingRate
            self.num_of_samples = int(reference_signal.numSamples)
        else:
            self.xt = None
        
        self.get_num_blocks()
        # Allowed sensor types in a dictionary
        self.get_allowed_sensors_dict()
        
        # Start an empty list of sensor properties
        self.sensor_list = []
        self.get_system_and_channels()
            
   
    def get_num_blocks(self,):
        """ Discover the number of blocks to measure and the number of samples to append.
        
        The number of blocks is given bu the intended signal length (num_of_samples)
        divided by the buffer_size.
        
        The plain division may give you a non-integer number_of_blocks. 
        In that case, the number_of_blocks is rounded to ceinling and the measured signals
        will be a bit longer. Then, the extra samples are tossed out.
        """
        self.num_of_blocks = int(np.ceil(self.num_of_samples/self.buffer_size))
        self.num_of_samples2toss = self.num_of_blocks * self.buffer_size - self.num_of_samples
    
    def get_allowed_sensors_dict(self,):
        """ Creates a dictionary with some important key-words and properties
        
        The key-words refer to allowed sensor types, and the properties are used
        when setting up the measurement.
        
        ----------
        The allowed sensors are one of the folowing: 'voltage' [V],
        'microphone' [Pa], 'force' [N], 'accelerometer' [m/s^2] or 'velocity' [m/s].
        We can add more later. Note the use of non-capital letters.
        sensitivity_units_SI_factor is used to convert sensitivity unitis to SI units
            
        """
        self.allowed_sensors = {
            'voltage' : {'sensor_units' : nidaqmx.constants.VoltageUnits.VOLTS,
                         'sensitivity_units' : 'Volt (NI default)',
                         'range_explained': 'min/max you expect to measure',
                         'sensitivity_units_SI_factor' : 1,
                         'range_scale_dB': False},
            'microphone' : {'sensor_units' : nidaqmx.constants.SoundPressureUnits.PA,
                         'sensitivity_units' : 'mV/Pa (NI default)',
                         'range_explained': 'Maximum instantaneous SPL you expect to measure.',
                         'sensitivity_units_SI_factor' : 0.001,
                         'range_scale_dB': True},
            'force' : {'sensor_units' : nidaqmx.constants.ForceUnits.NEWTONS,
                         'sensitivity_units' : nidaqmx.constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
                         'range_explained': 'min/max you expect to measure',
                         'sensitivity_units_SI_factor' : 0.001,
                         'range_scale_dB': False},
            'accelerometer' : {'sensor_units' : nidaqmx.constants.AccelUnits.METERS_PER_SECOND_SQUARED,
                         'sensitivity_units' : nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                         'range_explained': 'min/max you expect to measure',
                         'sensitivity_units_SI_factor' : 0.001,
                         'range_scale_dB': False},
            'velocity' : {'sensor_units' : nidaqmx.constants.VelocityUnits.METERS_PER_SECOND,
                         'sensitivity_units' : nidaqmx.constants.VelocityIEPESensorSensitivityUnits.MILLIVOLTS_PER_MILLIMETER_PER_SECOND,
                         'range_explained': 'min/max you expect to measure',
                         'sensitivity_units_SI_factor' : 1,
                         'range_scale_dB': False}
            }
    
    def get_system_and_channels(self,):
        """ Get the system info and channels' names in a flattened list
        
        Generates the self.idx_ai and self.idx_ao with the channels' names of the
        analog inputs and outputs. The index in this list is your physical channles.
        For instance, analog input channel 0 will have the name in self.idx_ai[0].
        When setting up each sensor, you will need to specify in which physical channel
        the sensor is. The sensor's dictionary will fill in the name of the channel for you.

        Parameters
        ----------        
        """
        # Get system
        self.system = nidaqmx.system.System.local()
        # List of dictionaries with IDs of analog inputs and outputs
        self.channel_list = []
        for device in self.system.devices:
            self.channel_list.append({
                "ai": [channels_ai.name
                       for _, channels_ai in enumerate(device.ai_physical_chans)],
                "ao":  [channels_ao.name
                        for _, channels_ao in enumerate(device.ao_physical_chans)]
            })
        
        # Get list of lists with analog inputs and outputs
        idx_ai = []
        idx_ao = []
        for idx in range(len(self.channel_list)):
            if self.channel_list[idx]['ai'] != []:
                idx_ai.append(self.channel_list[idx]['ai'])
            if self.channel_list[idx]['ao'] != []:
                idx_ao.append(self.channel_list[idx]['ao'])
        
        # Flatten the lists of ai and ao
        self.idx_ai = [item for sublist in idx_ai for item in sublist]
        self.idx_ao = [item for sublist in idx_ao for item in sublist]
    
    def set_output_channels(self,  physical_channel_nums = [0],
                            ao_range = 10):
        """ Set the output channels

        Parameters
        ----------
        physical_channel_nums : list
            list of integers with the channels you want to play signal through
        ao_range : float
            range in V of the output channel
        """
        self.output_list = []
        for channel_num in physical_channel_nums:
            # Get system channel name from self.idx_ai
            channel_name = self.idx_ao[channel_num]
            # create sensor's dictionary
            output_dict = {'sensor_type': 'voltage',
                           'physical_channel' : channel_num,
                           'channel_name' : channel_name,
                           'ao_range' : ao_range,
                           'out_units' : nidaqmx.constants.VoltageUnits.VOLTS}
            # append it to the list of sensors in your measurement
            self.output_list.append(output_dict)
      
    def set_sensor_properties(self, sensor_type = 'microphone', physical_channel_num = 0,
                              sensor_current = None, sensitivity = 1000.0,
                              ai_range = 1.0):
        """ Create a dictionary of the sensor properties and apped it to a list of sensors
        
        Parameters
        ----------
        sensor_type : str
            The type of sensor. It can be one of the folowing: 'voltage' [V],
            'microphone' [Pa], 'force' [N], 
            'accelerometer' [m/s^2] or 'velocity' [m/s].
            We can add more later. Note the use of non-capital letters.
        physical_channel_num : int
            The channel number in which you connected the sensor
        sensor_current : float
            Supply current in [A] to IEPE sensors (only applies to them)
        sensitivity:
            Sensitivity of the sensor in [mV/SensorUnits]. [SensorUnits] will be fetched 
            depending on the type
        ai_range : float
            The voltage range of your measurement in [V]. Your scale goes from
            -ai_range to +ai_range
        """
        # sensor_units, sensitivity_units = self.get_units(sensor_type = sensor_type)
        # Get sensor units and sensitivity units in a dictionary with keys in a list
        sensor_sens_units, keys_list = self.get_units(sensor_type = sensor_type)
        # Get system channel name from self.idx_ai
        channel_name = self.idx_ai[physical_channel_num]
        
        if sensor_sens_units[keys_list[4]]:
            pmax_Pa = (20e-6) * 10**(ai_range/20)
            ai_range_in_volt = pmax_Pa * sensitivity * sensor_sens_units[keys_list[3]]
        else:
            ai_range_in_volt = ai_range * sensitivity * sensor_sens_units[keys_list[3]]
        
        # create sensor's dictionary
        sensor_dict = {'sensor_type': sensor_type,
                       'physical_channel' : physical_channel_num,
                       'channel_name' : channel_name,
                       'ai_range' : ai_range,
                       'ai_range_in_volt' : ai_range_in_volt,
                       'range_explained' : sensor_sens_units[keys_list[2]],
                       'sensor_units' : sensor_sens_units[keys_list[0]],
                       'sensitivity' : sensitivity,
                       'sensitivity_SI' : sensitivity * sensor_sens_units[keys_list[3]],
                       'sensitivity_units' : sensor_sens_units[keys_list[1]],
                       'sensor_current' : sensor_current}
        # append it to the list of sensors in your measurement
        self.sensor_list.append(sensor_dict)
    
    def get_units(self, sensor_type = 'microphone'):
        """ Get sensor and sensitivity units from sensor type
        
        Parameters
        ----------
        sensor_type : str
            The type of sensor. It can be one of the folowing: 'voltage' [V],
            'microphone' [Pa], 'force' [N], 
            'accelerometer' [m/s^2] or 'velocity' [m/s].
            We can add more later. Note the use of non-capital letters.
        """
        if sensor_type in self.allowed_sensors:
            sensor_sens_units = self.allowed_sensors[sensor_type]
        else:
            sensor_sens_units = self.allowed_sensors['voltage']
        keys_list = list(sensor_sens_units.keys())
        return sensor_sens_units, keys_list
          
    def get_rec_channel(self, read_task_ai_channels, sensor):
        """ Get channel properties. We can expand this method to include more
            sensors in the future
        """
        # allowed_sensors = list(self.allowed_sensors.keys())
        if sensor['sensor_type'] == 'voltage' or sensor['sensor_type'] not in self.allowed_sensors:
            read_task_ai_channels.add_ai_voltage_chan(
                physical_channel = sensor['channel_name'], 
                min_val = -sensor['ai_range'], 
                max_val = sensor['ai_range'],
                units = sensor['sensor_units'])

        elif sensor['sensor_type'] == 'microphone':
            read_task_ai_channels.add_ai_microphone_chan(
                physical_channel = sensor['channel_name'], 
                units = sensor['sensor_units'], 
                mic_sensitivity = sensor['sensitivity'],
                max_snd_press_level = sensor['ai_range'],
                current_excit_val = sensor['sensor_current'])
        
        elif sensor['sensor_type'] == 'force':
            read_task_ai_channels.add_ai_force_iepe_chan(
                physical_channel = sensor['channel_name'],
                min_val = -sensor['ai_range'], 
                max_val = sensor['ai_range'],
                units = sensor['sensor_units'],
                sensitivity = sensor['sensitivity'],
                sensitivity_units = sensor['sensitivity_units'],
                current_excit_val = sensor['sensor_current'])
            
        elif sensor['sensor_type'] == 'accelerometer':
            read_task_ai_channels.add_ai_accel_chan(
                physical_channel = sensor['channel_name'],
                min_val = -sensor['ai_range'], 
                max_val = sensor['ai_range'],
                units = sensor['sensor_units'],
                sensitivity = sensor['sensitivity'],
                sensitivity_units = sensor['sensitivity_units'],
                current_excit_val = sensor['sensor_current'])
            
        elif sensor['sensor_type'] == 'velocity':
            read_task_ai_channels.add_ai_velocity_iepe_chan(
                physical_channel = sensor['channel_name'],
                min_val = -sensor['ai_range'], 
                max_val = sensor['ai_range'],
                units = sensor['sensor_units'],
                sensitivity = sensor['sensitivity'],
                sensitivity_units = sensor['sensitivity_units'],
                current_excit_val = sensor['sensor_current'])
        
        return read_task_ai_channels
    
    def get_out_channel(self, write_task_ao_channels, output):
        """ Get channel properties. We can expand this method to include more
            sensors in the future
        """
        
        write_task_ao_channels.add_ao_voltage_chan(
            physical_channel = output['channel_name'], 
            min_val = -output['ao_range'], 
            max_val = output['ao_range'],
            units = output['out_units'])
        
    def estimate_sensor_dBFS(self, rec_signal, sensitivity, voltage_range = 5):
        """ Estimate the level of the signal in dBFS
        
        This is done by taking the maximum value of the absolute of the signal,
        correcting the signal's scale to volts, and taking the ai_range of the 
        channel into account.
        """
        # Max value in sensor units
        max_sig_val_units = np.amax(np.abs(rec_signal))
        # Max value in sensor units
        max_sig_val_volt = max_sig_val_units * sensitivity
        # dBFS
        dBFS = round(20*np.log10(max_sig_val_volt/voltage_range), 2)
        return dBFS
    
    def estimate_allsensors_dBFS(self, values_read):
        """ Estimate the level of the signal in dBFS
        """
        dBFS_list = []
        print('Acqusition ended:')
        for ids, sensor in enumerate(self.sensor_list):
            dBFS = self.estimate_sensor_dBFS(values_read[ids,:], sensor['sensitivity_SI'],
                                             voltage_range = sensor['ai_range_in_volt'])
            dBFS_list.append(dBFS)
            print(sensor['sensor_type'] + ' max value: {} dBFS.'.format(dBFS))
    
    def recording_loop(self, read_task):
        """ This function contains the recording loop only
        
        This loop can be used in multiple functionalities
        """
        # Starting the data acquisition
        index_counter = 0
        values_read = np.zeros((len(self.sensor_list), 
                                self.num_of_blocks*self.buffer_size))
        
        # Recording loop (block by Block)
        print('Acquiring signals....')
        for iblock in range(self.num_of_blocks):
            # Read the current block
            current_buffer_raw = read_task.read(
                number_of_samples_per_channel = self.buffer_size)
            # Store data in a numpy array: size - n_in_channels x len(xt)
            values_read[:, index_counter:index_counter+self.buffer_size] = \
                current_buffer_raw
            # update index
            index_counter += self.buffer_size
        # discard last samples from extra buffer
        recorded_signals = values_read[:,:self.num_of_samples]
        # dBFS estimation
        self.estimate_allsensors_dBFS(recorded_signals)
        return recorded_signals
        
    def config_timing(self, task):
        """ config timing
        """
        task.timing.cfg_samp_clk_timing(
            rate = self.fs,
            # source = "OnboardClock",# '/cDAQ1Mod1/ai/SampleClock', #
            sample_mode = AcquisitionType.CONTINUOUS)

    def config_regeneation(self, task):
        """ config regeneration
        """
        task.out_stream.regen_mode = \
            nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
    
    def rec(self, ):
        """ record only signals

        Parameters
        ----------
        """
        Coupling.AC
        with nidaqmx.Task() as read_task:
            # Write the input channels
            for sensor in self.sensor_list:
                self.get_rec_channel(read_task.ai_channels, sensor)
            # Configure timing, controls and start task
            self.config_timing(read_task)
            read_task.control(TaskMode.TASK_COMMIT)
            read_task.start()
            # recording loop
            recorded_signals = self.recording_loop(read_task)
            # Pass to pytta object 
            meas_sig_obj = pytta.classes.SignalObj(signalArray = recorded_signals, 
                domain='time', samplingRate = self.fs)
            return meas_sig_obj

    def play(self, ):
        """ Play signal

        Parameters
        ----------
        """
        Coupling.AC
        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            # Get output voltage channel specs           
            self.get_out_channel(write_task.ao_channels, self.output_list[0])
            # Regeneration mode
            self.config_regeneation(write_task)
            # timing of output
            self.config_timing(write_task)
            # Get recording channel specs - we need this to play - some god knows why.
            dummy_reading_dict = {'sensor_type': 'voltage',
                           'physical_channel' : 0,
                           'channel_name' : self.idx_ai[0],
                           'ai_range' : 5,
                           'sensor_units' : nidaqmx.constants.VoltageUnits.VOLTS}
            self.get_rec_channel(read_task.ai_channels, dummy_reading_dict)
            # timing of input
            self.config_timing(read_task)            
            # Write the signal on the output channel
            write_task.write(self.xt.timeSignal[:,0])
            # commit the tasks
            write_task.control(TaskMode.TASK_COMMIT)
            read_task.control(TaskMode.TASK_COMMIT)
            # start the tasks
            write_task.start()
            read_task.start()
            print('Playing signal (Reading nothing)...')
            for iblock in range(self.num_of_blocks):
                _ = read_task.read(number_of_samples_per_channel = self.buffer_size)    
                
    def play_rec(self, ):
        """ Play-rec signals

        Parameters
        ----------
        """
        Coupling.AC

        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            # write the out channels (voltage channel)
            for output in self.output_list:
                self.get_out_channel(write_task.ao_channels, output)            
            # Regeneration mode
            self.config_regeneation(write_task)
            # timing of output
            self.config_timing(write_task)
            # Write the input channels
            for sensor in self.sensor_list:
                self.get_rec_channel(read_task.ai_channels, sensor)
            # timing of input
            self.config_timing(read_task)
            
            # Write the signal on the output channels - now you have two outs
            output_signal_mtx = np.tile(self.xt.timeSignal[:,0], (len(self.output_list),1))
            # print(output_signal_mtx.shape)
            if output_signal_mtx.shape[0] == 1:
                write_task.write(output_signal_mtx.flatten())
            else:
                write_task.write(output_signal_mtx)
            # commit the tasks
            write_task.control(TaskMode.TASK_COMMIT)
            read_task.control(TaskMode.TASK_COMMIT)
            
            ##### FROM NI
            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                read_task.triggers.start_trigger.term)
            
            # start the tasks
            write_task.start()
            read_task.start()
            
            # recording loop
            recorded_signals = self.recording_loop(read_task)
            # Pass to pytta object 
            meas_sig_obj = pytta.classes.SignalObj(signalArray = recorded_signals, 
                domain='time', samplingRate = self.fs)
        return meas_sig_obj 
    
    # def set_output_channels(self,  out_channel_to_ni = 0, out_channel_to_amp = 3,
    #                         ao_range = 10):
    #     """ Set the output channels

    #     Parameters
    #     ----------
    #     out_channel_to_ni : int
    #         Channel index that you output and input back at NI
    #     out_channel_to_amp : int
    #         Channel index that you output and amplifier (ant then, loudspeaker)
    #     ao_range : float
    #         range in V of the output channel
    #     """
    #     self.out_channel_to_ni = out_channel_to_ni
    #     self.out_channel_to_amp = out_channel_to_amp
    #     self.out_channel_to_ni_name = self.idx_ao[0][self.out_channel_to_ni]
    #     self.out_channel_to_amp_name = self.idx_ao[0][self.out_channel_to_amp]
    #     self.ao_range = ao_range
    
    # def set_input_channels(self,  in_channel_ref = 0, in_channel_sensor = [1],
    #                        ai_range = 5, sensor_sens = 50, sensor_current = 0):
    #     """ Set the input channels

    #     Parameters
    #     ----------
    #     in_channel_ref : int
    #         Channel index in which you input your reference signal
    #     in_channel_sensor : list of integers
    #         Channels indexes in which you input your sensor signal
    #     ai_range : float
    #         range in V of the input channels
    #     sensor_sens : float
    #         sensor sensitivity in mV/[SI units] (e.g. V/Pa)
    #     """
    #     self.in_channel_ref = in_channel_ref
    #     self.in_channel_sensor = in_channel_sensor
    #     self.in_channel_ref_name = self.idx_ai[0][self.in_channel_ref]
    #     self.in_channel_sensor_names = []
    #     for id_in_sensor in range(len(self.in_channel_sensor)):
    #         self.in_channel_sensor_names.append(self.idx_ai[0][self.in_channel_sensor[id_in_sensor]])
    #     self.ai_range = ai_range
    #     self.sensor_sens = sensor_sens
    #     self.sensor_current = sensor_current
    
    # def get_output_task(self, ):
    #     """ Get output task for NI with two outs
    #     """
    #     Coupling.AC
    #     self.output_task = nidaqmx.Task() #new_task_name = 'outtask' + str(self.rn)
        
    #     self.output_task.ao_channels.add_ao_voltage_chan(
    #         self.out_channel_to_ni_name,
    #         max_val=self.ao_range,
    #         min_val=-self.ao_range)
        
    #     self.output_task.ao_channels.add_ao_voltage_chan(
    #         self.out_channel_to_amp_name,
    #         max_val=self.ao_range,
    #         min_val=-self.ao_range)
        
    #     # Regeneration mode
    #     self.output_task.out_stream.regen_mode = \
    #         nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
    #     # timing of output
    #     self.output_task.timing.cfg_samp_clk_timing(
    #         rate = self.xt.samplingRate,
    #         sample_mode = AcquisitionType.CONTINUOUS)
    #     # self.output_task.timing.cfg_samp_clk_timing(
    #     #     rate = self.xt.samplingRate,
    #     #     sample_mode = AcquisitionType.FINITE,
    #     #     samps_per_chan = self.buffer_size) #AcquisitionType.CONTINUOUS
        
        
    #     # Write the signal on the output channels - now you have two outs
    #     self.output_task.write(np.tile(self.xt.timeSignal[:,0], (2,1)))
        
    #     self.output_task.control(TaskMode.TASK_COMMIT)
    #     #self.output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/cDAQ1/ai/StartTrigger')
    #     #####################################
           
    #     # return output_task
    
    # def get_input_task(self, ):
    #     """ Get input task for NI. At this point - just microphones
        
    #     Parameters
    #     ----------
    #     input_type : str
    #         string with type of recording. Can be either 'mic' for microphone
    #         or 'voltage' to configure by pass measurement. Default or any str sets
    #         to 'mic'
    #     """
    #     #input_unit = SoundPressureUnits.PA
    #     # Max of sound card for dBFS
    #     # Coupling.AC
    #     #self.max_val_dbFS = self.input_dict['max_min_val'][1]
    #     # Instantiate NI object
    #     self.input_task = nidaqmx.Task() #new_task_name = 'intask' + str(self.rn)
        
    #     # Add the channels
    #     # 1 ref read channel
    #     self.input_task.ai_channels.add_ai_voltage_chan(
    #         self.in_channel_ref_name,
    #         max_val=self.ai_range,
    #         min_val=-self.ai_range)
        
    #     # sensor read channels
    #     for channel_name in self.in_channel_sensor_names:
    #         self.input_task.ai_channels.add_ai_microphone_chan(
    #             channel_name, 
    #             units = SoundPressureUnits.PA, 
    #             mic_sensitivity = self.sensor_sens,
    #             current_excit_val = self.sensor_current)
        
    #     # # Configure input signal
    #     # if input_type == 'mic':
    #     #     self.input_task.ai_channels.add_ai_microphone_chan(
    #     #         self.input_dict['terminal'], 
    #     #         units = SoundPressureUnits.PA, 
    #     #         mic_sensitivity = self.input_dict['mic_sens'],
    #     #         current_excit_val = self.input_dict['current_exc_sensor'])
    #     # elif input_type == 'voltage':
    #     #     self.input_task.ai_channels.add_ai_voltage_chan(
    #     #         self.input_dict['terminal'],
    #     #         min_val = self.input_dict['max_min_val'][0], 
    #     #         max_val = self.input_dict['max_min_val'][1],
    #     #         units = VoltageUnits.VOLTS)
    #     # else:
    #     #     self.input_task.ai_channels.add_ai_microphone_chan(
    #     #         self.input_dict['terminal'], 
    #     #         units = SoundPressureUnits.PA, 
    #     #         mic_sensitivity = self.input_dict['mic_sens'],
    #     #         current_excit_val = self.input_dict['current_exc_sensor'])
        
    #     # timing of input
    #     self.input_task.timing.cfg_samp_clk_timing(
    #         rate = self.xt.samplingRate,
    #         sample_mode = AcquisitionType.CONTINUOUS)
    #     # self.input_task.timing.cfg_samp_clk_timing(
    #     #     rate = self.xt.samplingRate,
    #     #     sample_mode = AcquisitionType.FINITE,
    #     #     samps_per_chan = self.buffer_size) #AcquisitionType.CONTINUOUS
        
    #     # commit
    #     self.input_task.control(TaskMode.TASK_COMMIT)
        
    #     # return input_task
    
    # def rec_newerversion(self, ):
    #     """ record only signals

    #     Parameters
    #     ----------
    #     """
    #     Coupling.AC

    #     with nidaqmx.Task() as read_task:
    #         # Write the input channels
    #         for sensor in self.sensor_list:
    #             self.get_rec_channel(read_task.ai_channels, sensor)
                
    #         # # timing of input
    #         # read_task.timing.cfg_samp_clk_timing(
    #         #     rate = self.fs,
    #         #     sample_mode = AcquisitionType.CONTINUOUS)
            
    #         # # commit the tasks
    #         # read_task.control(TaskMode.TASK_COMMIT)
            
    #         # # start the read task
    #         # read_task.start()
            
    #         self.config_timing(read_task)
    #         read_task.control(TaskMode.TASK_COMMIT)
    #         read_task.start()
            
    #         # recording loop
    #         recorded_signals = self.recording_loop(read_task)
    #         # # Starting the data acquisition
    #         # index_counter = 0
    #         # values_read = np.zeros((len(self.sensor_list), 
    #         #                         self.num_of_blocks*self.buffer_size))
            
    #         # # Recording loop (block by Block)
    #         # print('Recording sound in the environment...')
    #         # for iblock in range(self.num_of_blocks):
    #         #     # Read the current block
    #         #     current_buffer_raw = read_task.read(
    #         #         number_of_samples_per_channel = self.buffer_size)
    #         #     # Store data in a numpy array: size - n_in_channels x len(xt)
    #         #     values_read[:, index_counter:index_counter+self.buffer_size] = \
    #         #         current_buffer_raw
    #         #     # update index
    #         #     index_counter += self.buffer_size
                
    #         # Pass to pytta object 
    #         meas_sig_obj = pytta.classes.SignalObj(signalArray = recorded_signals, 
    #             domain='time', samplingRate = self.fs)
            
    #         # self.estimate_allsensors_dBFS(values_read)

    #         return meas_sig_obj
    
    
    # def play_rec_newerversion(self, ):
    #     """ Play-rec signals

    #     Parameters
    #     ----------
    #     """
    #     Coupling.AC

    #     with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
    #         # write the out channels (voltage channel)
    #         for output in self.output_list:
    #             self.get_out_channel(write_task.ao_channels, output)
    #         # # To NI
    #         # write_task.ao_channels.add_ao_voltage_chan(
    #         #     self.out_channel_to_ni_name,
    #         #     max_val=self.ao_range,
    #         #     min_val=-self.ao_range)
    #         # # To amp
    #         # write_task.ao_channels.add_ao_voltage_chan(
    #         #     self.out_channel_to_amp_name,
    #         #     max_val=self.ao_range,
    #         #     min_val=-self.ao_range)
    #         # Regeneration mode
            
    #         # Regeneration mode
    #         self.config_regeneation(write_task)
    #         # timing of output
    #         self.config_timing(write_task)
    #         # write_task.out_stream.regen_mode = \
    #         #     nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
    #         # # timing of output
    #         # write_task.timing.cfg_samp_clk_timing(
    #         #     rate = self.xt.samplingRate,
    #         #     sample_mode = AcquisitionType.CONTINUOUS)
            
    #         # Write the input channels
    #         for sensor in self.sensor_list:
    #             self.get_rec_channel(read_task.ai_channels, sensor)
    #         # # Write the input channels
    #         # # 1 ref read channel
    #         # read_task.ai_channels.add_ai_voltage_chan(
    #         #     self.in_channel_ref_name,
    #         #     max_val=self.ai_range,
    #         #     min_val=-self.ai_range)
            
    #         # # sensor read channels
    #         # for channel_name in self.in_channel_sensor_names:
    #         #     read_task.ai_channels.add_ai_microphone_chan(
    #         #         channel_name, 
    #         #         units = SoundPressureUnits.PA, 
    #         #         mic_sensitivity = self.sensor_sens,
    #         #         current_excit_val = self.sensor_current)
            
    #         # timing of input
    #         self.config_timing(read_task)
            
    #         # read_task.timing.cfg_samp_clk_timing(
    #         #     rate = self.xt.samplingRate,
    #         #     sample_mode = AcquisitionType.CONTINUOUS)
                
    #         # Write the signal on the output channels - now you have two outs
    #         write_task.write(np.tile(self.xt.timeSignal[:,0], (len(self.output_list),1)))
    #         # write_task.write(self.xt.timeSignal[:,0])
            
    #         # commit the tasks
    #         write_task.control(TaskMode.TASK_COMMIT)
    #         read_task.control(TaskMode.TASK_COMMIT)
            
    #         # start the tasks
    #         write_task.start()
    #         read_task.start()
            
    #         # recording loop
    #         recorded_signals = self.recording_loop(read_task)
            
    #         # # Starting the data acquisition/reproduction
    #         # # Initialization
    #         # index_counter = 0
    #         # values_read = np.zeros((len(self.in_channel_sensor_names)+1, len(self.xt.timeSignal[:,0])))      
  
    #         # # Main loop (we read blocks of the signal of self.buffer_size samples)
    #         # number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
    #         # print('Acqusition started')
    #         # for iblock in range(number_of_blocks):
    #         #     # Read the current block
    #         #     current_buffer_raw = read_task.read(
    #         #         number_of_samples_per_channel = self.buffer_size)
    #         #     # Store data in a numpy array: size - n_in_channels x len(xt)
    #         #     values_read[:, index_counter:index_counter+self.buffer_size] = \
    #         #         current_buffer_raw
    #         #     # update index
    #         #     index_counter += self.buffer_size
            
            
            
    #         # Pass to pytta object 
    #         meas_sig_obj = pytta.classes.SignalObj(signalArray = recorded_signals, 
    #             domain='time', samplingRate = self.fs)
    #         # # Estimate dBFS of In and out
    #         # dBFS_ref = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,0]))/self.ai_range), 2)
    #         # dBFS_mic = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,1:]*self.sensor_sens/1000))/self.ai_range), 2)
    #         # print('Acqusition ended: Max_In_Val = {} dBFS, Max_Out_Val = {} dBFS'.format(dBFS_ref, dBFS_mic))
    #     return meas_sig_obj 
    
    
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
        
    # def initializer(self, ):
    #     """ For some reason unkown to mankind, the very first recording with a mic is bad

    #     I`ll do some dummy recording.
        
    #     Parameters
    #     ----------
    #     """
    #     number_of_dummy_blocks = int(2**20/self.buffer_size)
    #     self.rec(dummy_rec = True, number_of_dummy_blocks = number_of_dummy_blocks)
    
    # def rec(self, dummy_rec = False, number_of_dummy_blocks = 2):
    #     """ rec signals

    #     Parameters
    #     ----------
    #     """
    #     Coupling.AC

    #     with nidaqmx.Task() as read_task:
    #         # Write the input channels
    #         # sensor read channels
    #         # sensor read channels
    #         for channel_name in self.in_channel_sensor_names:
    #             read_task.ai_channels.add_ai_microphone_chan(
    #                 channel_name, 
    #                 units = SoundPressureUnits.PA, 
    #                 mic_sensitivity = self.sensor_sens,
    #                 current_excit_val = self.sensor_current)
            
    #         # timing of input
    #         read_task.timing.cfg_samp_clk_timing(
    #             rate = self.xt.samplingRate,
    #             sample_mode = AcquisitionType.CONTINUOUS)
            
    #         # commit the tasks
    #         # write_task.control(TaskMode.TASK_COMMIT)
    #         read_task.control(TaskMode.TASK_COMMIT)
            
    #         # start the read task
    #         read_task.start()
    #         # Starting the data acquisition
    #         # Initialization
    #         index_counter = 0
    #         # values_read = np.zeros((len(self.in_channel_sensor_names), len(self.xt.timeSignal[:,0])))      
  
    #         # Main loop (we read blocks of the signal of self.buffer_size samples)
    #         if dummy_rec:
    #             number_of_blocks = int(number_of_dummy_blocks)
    #             values_read = np.zeros((len(self.in_channel_sensor_names), self.buffer_size*number_of_blocks))
    #             print('Dummy recording - it is long, but necessary')
    #         else:
    #             number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
    #             values_read = np.zeros((len(self.in_channel_sensor_names), len(self.xt.timeSignal[:,0])))
    #             print('Recording sound in the environment')

            
    #         for iblock in range(number_of_blocks):
    #             # Read the current block
    #             current_buffer_raw = read_task.read(
    #                 number_of_samples_per_channel = self.buffer_size)
    #             # Store data in a numpy array: size - n_in_channels x len(xt)
    #             values_read[:, index_counter:index_counter+self.buffer_size] = \
    #                 current_buffer_raw
    #             # update index
    #             index_counter += self.buffer_size
            
    #         # Pass read data to pytta signal objects  
    #         # sensor_obj_list = []
    #         # for channel_num in range(len(self.in_channel_sensor_names)):
    #         #     sensor_obj = pytta.classes.SignalObj(signalArray = values_read[channel_num,:], 
    #         #         domain='time', freqMin = self.xt.freqMin, 
    #         #         freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
    #         #     sensor_obj_list.append(sensor_obj)
                        
    #         # # Estimate dBFS of In and out
    #         # dBFS_mic = round(20*np.log10(np.amax(np.abs(sensor_obj.timeSignal*self.sensor_sens/1000))/self.ai_range), 2)
            
    #         meas_sig_obj = pytta.classes.SignalObj(signalArray = values_read, 
    #             domain='time', freqMin = self.xt.freqMin, 
    #             freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
    #         # Estimate dBFS of In and out
    #         dBFS_mic = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal*self.sensor_sens/1000))/self.ai_range), 2)
            
    #         print('Acqusition ended: Max_Out_Val = {} dBFS'.format(dBFS_mic))
    #         return meas_sig_obj
    
    # def play_rec2(self, ):
    #     """ Play-rec signals

    #     Parameters
    #     ----------
    #     """
    #     Coupling.AC
 
    #     # start the tasks
    #     #self.get_output_task()
    #     # self.get_input_task()
    #     self.output_task.start()
    #     self.input_task.start()
    #     # Starting the data acquisition/reproduction
    #     # Initialization
    #     index_counter = 0
    #     values_read = np.zeros((len(self.in_channel_sensor_names)+1, len(self.xt.timeSignal[:,0])))      
  
    #     # Main loop (we read blocks of the signal of self.buffer_size samples)
    #     number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
    #     print('Acqusition started')
    #     for iblock in range(number_of_blocks):
    #         # Read the current block
    #         current_buffer_raw = self.input_task.read(
    #             number_of_samples_per_channel = self.buffer_size) # , timeout = 2*round(self.buffer_size/self.fs, 2)
    #         # Store data in a numpy array: size - n_in_channels x len(xt)
    #         values_read[:, index_counter:index_counter+self.buffer_size] = \
    #             current_buffer_raw
    #         # update index
    #         index_counter += self.buffer_size
        
    #     self.input_task.stop()
    #     # self.input_task.close()
    #     #self.output_task.stop()
    #     #self.output_task.close()
        
        
    #     meas_sig_obj = pytta.classes.SignalObj(signalArray = values_read, 
    #         domain='time', freqMin = self.xt.freqMin, 
    #         freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
    #     # Estimate dBFS of In and out
    #     dBFS_ref = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,0]))/self.ai_range), 2)
    #     dBFS_mic = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,1:]*self.sensor_sens/1000))/self.ai_range), 2)
    #     print('Acqusition ended: Max_In_Val = {} dBFS, Max_Out_Val = {} dBFS'.format(dBFS_ref, dBFS_mic))
    #     return meas_sig_obj #ref_obj, sensor_obj_list

        
        