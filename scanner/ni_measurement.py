# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:31:35 2024

@author: Admin
"""

# general imports
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import scipy.io as io
from scipy.signal import windows, resample, chirp
from datetime import datetime

# Pytta imports
import pytta
# from pytta.generate import sweep
# from pytta.classes import SignalObj, FRFMeasure
# from pytta import ImpulsiveResponse, save, merge

# Arduino imports

# NI imports
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode, RegenerationMode, Coupling
from nidaqmx.constants import SoundPressureUnits, VoltageUnits
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
    
    def __init__(self, reference_sweep = None, fs = 51200, buffer_size = None):
        """

        Parameters
        ----------
        reference_sweep : pytta obj
            reference sweep to be executed
        fs : int
            sampling rate
        buffer_size : int or None
            The size of your buffer on playback. It should be a power of 2 (since your playback
            signal has size of a power of 2. This means that a integer number of blocks are played-back)
        
        """
        # folder checking
        self.fs = fs
        if reference_sweep is None:
            print('You did not specify any sweep. I am generating a default one for you.')
            self.xt = pytta.generate.sweep(freqMin = 100,
              freqMax = 10000, samplingRate = self.fs,
              fftDegree = 18, startMargin = 0.1,
              stopMargin = 0.5,
              method = 'logarithmic', windowing='hann')
        else:
            self.xt = reference_sweep
        
        if buffer_size is None:
            self.buffer_size = 2**10
        elif np.log2(buffer_size).is_integer():
            self.buffer_size = buffer_size
        else:
            self.buffer_size = int(2**np.floor(np.log2(buffer_size)))
        
    def get_system_and_channels(self,):
        """ Get the system info and channels list

        Parameters
        ----------        
        """
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
            
        self.idx_ai = []
        self.idx_ao = []
        for idx in range(len(self.channel_list)):
            if self.channel_list[idx]['ai'] != []:
                self.idx_ai.append(self.channel_list[idx]['ai'])
            if self.channel_list[idx]['ao'] != []:
                self.idx_ao.append(self.channel_list[idx]['ao'])
    
    def set_output_channels(self,  out_channel_to_ni = 0, out_channel_to_amp = 3,
                            ao_range = 10):
        """ Set the output channels

        Parameters
        ----------
        out_channel_to_ni : int
            Channel index that you output and input back at NI
        out_channel_to_amp : int
            Channel index that you output and amplifier (ant then, loudspeaker)
        ao_range : float
            range in V of the output channel
        """
        self.out_channel_to_ni = out_channel_to_ni
        self.out_channel_to_amp = out_channel_to_amp
        self.out_channel_to_ni_name = self.idx_ao[0][self.out_channel_to_ni]
        self.out_channel_to_amp_name = self.idx_ao[0][self.out_channel_to_amp]
        self.ao_range = ao_range
    
    def set_input_channels(self,  in_channel_ref = 0, in_channel_sensor = [1],
                           ai_range = 5, sensor_sens = 50, sensor_current = 0):
        """ Set the input channels

        Parameters
        ----------
        in_channel_ref : int
            Channel index in which you input your reference signal
        in_channel_sensor : list of integers
            Channels indexes in which you input your sensor signal
        ai_range : float
            range in V of the input channels
        sensor_sens : float
            sensor sensitivity in mV/[SI units] (e.g. V/Pa)
        """
        self.in_channel_ref = in_channel_ref
        self.in_channel_sensor = in_channel_sensor
        self.in_channel_ref_name = self.idx_ai[0][self.in_channel_ref]
        self.in_channel_sensor_names = []
        for id_in_sensor in range(len(self.in_channel_sensor)):
            self.in_channel_sensor_names.append(self.idx_ai[0][self.in_channel_sensor[id_in_sensor]])
        self.ai_range = ai_range
        self.sensor_sens = sensor_sens
        self.sensor_current = sensor_current
        
    def initializer(self, ):
        """ For some reason unkown to mankind, the very first recording with a mic is bad

        I`ll do some dummy recording.
        
        Parameters
        ----------
        """
        number_of_dummy_blocks = int(2**20/self.buffer_size)
        self.rec(dummy_rec = True, number_of_dummy_blocks = number_of_dummy_blocks)
        
    def play(self, ):
        """ Play signal through amplifier - just to listen

        Parameters
        ----------
        """
        Coupling.AC
        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            # write the out channel (voltage channel)
            write_task.ao_channels.add_ao_voltage_chan(
                self.out_channel_to_amp_name,
                max_val=self.ao_range,
                min_val=-self.ao_range)
            # Regeneration mode
            write_task.out_stream.regen_mode = \
                nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
            # timing of output
            write_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode=AcquisitionType.CONTINUOUS)
            
            # You need to write a read task to play the signal (reason - unknown)
            read_task.ai_channels.add_ai_voltage_chan(
                self.in_channel_ref_name,
                max_val=self.ai_range,
                min_val=-self.ai_range)
            # timing of input
            read_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
            
            # Write the signal on the output channel
            write_task.write(self.xt.timeSignal[:,0])
            # commit the tasks
            write_task.control(TaskMode.TASK_COMMIT)
            read_task.control(TaskMode.TASK_COMMIT)
            # start the tasks
            write_task.start()
            read_task.start()
            # write_task.stop()
            # write_task.close()
            # Main loop
            number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
            # while timeCounter < number_of_blocks:
            print('Playing signal (Reading nothing)')
            for iblock in range(number_of_blocks):
                _ = read_task.read(
                    number_of_samples_per_channel = self.buffer_size)
                
    def rec(self, dummy_rec = False, number_of_dummy_blocks = 2):
        """ rec signals

        Parameters
        ----------
        """
        Coupling.AC

        with nidaqmx.Task() as read_task:
            # Write the input channels
            # sensor read channels
            # sensor read channels
            for channel_name in self.in_channel_sensor_names:
                read_task.ai_channels.add_ai_microphone_chan(
                    channel_name, 
                    units = SoundPressureUnits.PA, 
                    mic_sensitivity = self.sensor_sens,
                    current_excit_val = self.sensor_current)
            
            # timing of input
            read_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
            
            # commit the tasks
            # write_task.control(TaskMode.TASK_COMMIT)
            read_task.control(TaskMode.TASK_COMMIT)
            
            # start the read task
            read_task.start()
            # Starting the data acquisition
            # Initialization
            index_counter = 0
            # values_read = np.zeros((len(self.in_channel_sensor_names), len(self.xt.timeSignal[:,0])))      
  
            # Main loop (we read blocks of the signal of self.buffer_size samples)
            if dummy_rec:
                number_of_blocks = int(number_of_dummy_blocks)
                values_read = np.zeros((len(self.in_channel_sensor_names), self.buffer_size*number_of_blocks))
                print('Dummy recording - it is long, but necessary')
            else:
                number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
                values_read = np.zeros((len(self.in_channel_sensor_names), len(self.xt.timeSignal[:,0])))
                print('Recording sound in the environment')

            
            for iblock in range(number_of_blocks):
                # Read the current block
                current_buffer_raw = read_task.read(
                    number_of_samples_per_channel = self.buffer_size)
                # Store data in a numpy array: size - n_in_channels x len(xt)
                values_read[:, index_counter:index_counter+self.buffer_size] = \
                    current_buffer_raw
                # update index
                index_counter += self.buffer_size
            
            # Pass read data to pytta signal objects  
            # sensor_obj_list = []
            # for channel_num in range(len(self.in_channel_sensor_names)):
            #     sensor_obj = pytta.classes.SignalObj(signalArray = values_read[channel_num,:], 
            #         domain='time', freqMin = self.xt.freqMin, 
            #         freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            #     sensor_obj_list.append(sensor_obj)
                        
            # # Estimate dBFS of In and out
            # dBFS_mic = round(20*np.log10(np.amax(np.abs(sensor_obj.timeSignal*self.sensor_sens/1000))/self.ai_range), 2)
            
            meas_sig_obj = pytta.classes.SignalObj(signalArray = values_read, 
                domain='time', freqMin = self.xt.freqMin, 
                freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            # Estimate dBFS of In and out
            dBFS_mic = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal*self.sensor_sens/1000))/self.ai_range), 2)
            
            print('Acqusition ended: Max_Out_Val = {} dBFS'.format(dBFS_mic))
            return meas_sig_obj
                
    def play_rec(self, ):
        """ Play-rec signals

        Parameters
        ----------
        """
        Coupling.AC

        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            # write the out channels (voltage channel) - on play-rec you will need 2
            # One goes to the reference input channel of NI, the other to amp
            # To NI
            write_task.ao_channels.add_ao_voltage_chan(
                self.out_channel_to_ni_name,
                max_val=self.ao_range,
                min_val=-self.ao_range)
            # To amp
            write_task.ao_channels.add_ao_voltage_chan(
                self.out_channel_to_amp_name,
                max_val=self.ao_range,
                min_val=-self.ao_range)
            # Regeneration mode
            write_task.out_stream.regen_mode = \
                nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
            # timing of output
            write_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
            
            # Write the input channels
            # 1 ref read channel
            read_task.ai_channels.add_ai_voltage_chan(
                self.in_channel_ref_name,
                max_val=self.ai_range,
                min_val=-self.ai_range)
            
            # sensor read channels
            for channel_name in self.in_channel_sensor_names:
                read_task.ai_channels.add_ai_microphone_chan(
                    channel_name, 
                    units = SoundPressureUnits.PA, 
                    mic_sensitivity = self.sensor_sens,
                    current_excit_val = self.sensor_current)
            
            # timing of input
            read_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
                
            # Write the signal on the output channels - now you have two outs
            write_task.write(np.tile(self.xt.timeSignal[:,0], (2,1)))
            # write_task.write(self.xt.timeSignal[:,0])
            
            # commit the tasks
            write_task.control(TaskMode.TASK_COMMIT)
            read_task.control(TaskMode.TASK_COMMIT)
            
            # start the tasks
            write_task.start()
            read_task.start()
            # Starting the data acquisition/reproduction
            # Initialization
            index_counter = 0
            values_read = np.zeros((len(self.in_channel_sensor_names)+1, len(self.xt.timeSignal[:,0])))      
  
            # Main loop (we read blocks of the signal of self.buffer_size samples)
            number_of_blocks = int(len(self.xt.timeSignal[:,0])/self.buffer_size)
            print('Acqusition started')
            for iblock in range(number_of_blocks):
                # Read the current block
                current_buffer_raw = read_task.read(
                    number_of_samples_per_channel = self.buffer_size)
                # Store data in a numpy array: size - n_in_channels x len(xt)
                values_read[:, index_counter:index_counter+self.buffer_size] = \
                    current_buffer_raw
                # update index
                index_counter += self.buffer_size
            
            # Pass read data to pytta signal objects
            # ref_obj = pytta.classes.SignalObj(signalArray = values_read[0,:], 
            #     domain='time', freqMin = self.xt.freqMin, 
            #     freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            
            # sensor_obj_list = []
            # for channel_num in range(len(self.in_channel_sensor_names)):
            #     sensor_obj = pytta.classes.SignalObj(signalArray = values_read[channel_num + 1,:], 
            #         domain='time', freqMin = self.xt.freqMin, 
            #         freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            #     sensor_obj_list.append(sensor_obj)
            # Estimate dBFS of In and out
            # dBFS_ref = round(20*np.log10(np.amax(np.abs(ref_obj.timeSignal))/self.ai_range), 2)
            # dBFS_mic = round(20*np.log10(np.amax(np.abs(sensor_obj.timeSignal*self.sensor_sens/1000))/self.ai_range), 2)
            # print('Acqusition ended: Max_In_Val = {} dBFS, Max_Out_Val = {} dBFS'.format(dBFS_ref, dBFS_mic))
            
            meas_sig_obj = pytta.classes.SignalObj(signalArray = values_read, 
                domain='time', freqMin = self.xt.freqMin, 
                freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            # Estimate dBFS of In and out
            dBFS_ref = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,0]))/self.ai_range), 2)
            dBFS_mic = round(20*np.log10(np.amax(np.abs(meas_sig_obj.timeSignal[:,1:]*self.sensor_sens/1000))/self.ai_range), 2)
            print('Acqusition ended: Max_In_Val = {} dBFS, Max_Out_Val = {} dBFS'.format(dBFS_ref, dBFS_mic))
        return meas_sig_obj #ref_obj, sensor_obj_list

        
        