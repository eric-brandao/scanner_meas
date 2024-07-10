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
    
    def __init__(self, reference_sweep = None, fs = 51200):
        """

        Parameters
        ----------
        reference_sweep : pytta obj
            reference sweep to be executed
        fs : int
            sampling rate
        
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
        
    def play(self,):
        """ Play signal through amplifier - just to listen

        Parameters
        ----------
        """
        Coupling.AC
        with nidaqmx.Task() as write_task:
            # print(write_task)
            # Create Voltage channel to play
            # print("{},{},{}".format(self.out_channel_to_amp_name, self.ao_range, -self.ao_range))
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
            # Write the signal
            write_task.write(self.xt.timeSignal[:,0])
            # commit
            write_task.control(TaskMode.TASK_COMMIT)
            # start, stop, close
            write_task.start()
            write_task.stop()
            # write_task.close()
            
    def play_rec(self, buffer_size = 2**12):
        """ Play-rec signals

        Parameters
        ----------
        """
        Coupling.AC

        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            # Write the output channels
            # To ref
            write_task.ao_channels.add_ao_voltage_chan(
                self.out_channel_to_ni_name,
                max_val=self.ao_range,
                min_val=-self.ao_range)
            
            # To amp
            write_task.ao_channels.add_ao_voltage_chan(
                self.out_channel_to_amp_name,
                max_val=self.ao_range,
                min_val=-self.ao_range)
            
            # Write the input channels
            # ref read channel
            read_task.ai_channels.add_ai_voltage_chan(
                self.in_channel_ref_name,
                max_val=self.ai_range,
                min_val=-self.ai_range)
            
            # sensor read channel
            read_task.ai_channels.add_ai_microphone_chan(
                self.in_channel_sensor_names[0], 
                units = SoundPressureUnits.PA, 
                mic_sensitivity = self.sensor_sens,
                current_excit_val = self.sensor_current)
            
            read_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
    
            
            write_task.out_stream.regen_mode = \
                nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
            write_task.timing.cfg_samp_clk_timing(
                rate = self.xt.samplingRate,
                sample_mode = AcquisitionType.CONTINUOUS)
            
            # write_task.write(self.xt.timeSignal[:,0])
            write_task.write(np.tile(self.xt.timeSignal[:,0], (2,1)))
            # write_task.write([self.xt.timeSignal[:,0], self.xt.timeSignal[:,0]])

            # if idx_ao:
            write_task.control(TaskMode.TASK_COMMIT)
            # if idx_ai:
            read_task.control(TaskMode.TASK_COMMIT)
            # if idx_ao:
            write_task.start()
            # if idx_ai:
            read_task.start()
    
            # Starting the data acquisition/reproduction
    
            # Initialization
            time_counter = 0
            # blockidx = 0
            values_read = np.zeros((2, len(self.xt.timeSignal[:,0])))
            
            # current_buffer = np.zeros((1, int(buffer_size)))
            # previous_buffer = np.zeros((1, int(buffer_size)))
  
            # Main loop
            # if idx_ai:
            number_of_blocks = int(len(self.xt.timeSignal[:,0])/buffer_size)
            # pbar_ai = tqdm(total = number_of_blocks, desc = 'Acquiring signals')
            # while timeCounter < number_of_blocks:
            print('Acqusition started')
            for iblock in range(number_of_blocks):
                # The current read buffer - a list variable
                current_buffer_raw = read_task.read(
                    number_of_samples_per_channel = buffer_size)
                # print(type(current_buffer_raw[0][0]))
                # This is the variable that stores the data for saving
                values_read[:, time_counter:time_counter+buffer_size] = \
                    current_buffer_raw 
                
                time_counter += buffer_size
            # Estimate dBFS
            dBFS = round(20*np.log10((np.amax(np.abs(values_read)*(self.sensor_sens/1000)))/self.ai_range), 2)
            print('Acqusition ended: {} dBFS'.format(dBFS))
            # Pass to pytta
            ref_obj = pytta.classes.SignalObj(signalArray = values_read[1,:], 
                domain='time', freqMin = self.xt.freqMin, 
                freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            
            sensor_obj = pytta.classes.SignalObj(signalArray = values_read[0,:], 
                domain='time', freqMin = self.xt.freqMin, 
                freqMax = self.xt.freqMax, samplingRate = self.xt.samplingRate)
            return sensor_obj, ref_obj

        
        