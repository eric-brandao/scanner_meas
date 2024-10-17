# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:45:32 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:07:17 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from ni_measurement import NIMeasurement

import pytta
import time

#%% Just record input
fs = 51200
ni_rec = NIMeasurement(fs = 51200, buffer_size = 2**10, fft_degree = None, time_length = 3)

ni_rec.set_sensor_properties(sensor_type = 'voltage', physical_channel_num = 0, sensitivity = 1, 
                             ai_range = 2)

ni_rec.set_sensor_properties(sensor_type = 'microphone', physical_channel_num = 6,
                          sensor_current = 4e-3, sensitivity = 54.8, ai_range = 130)

rec_signal = ni_rec.rec()
rec_signal.plot_time();

#%% Generate a sweep to play and play rec
fs = 51200
freq_min = 100
freq_max = 10000
fft_degree = 18
start_margin = 0.1
stop_margin = 0.5

xt = pytta.generate.sweep(freqMin = freq_min,
  freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, startMargin = start_margin,
  stopMargin = stop_margin, method = 'logarithmic', windowing='hann')

#%% Just play signal
ni_play = NIMeasurement(fs = 51200, buffer_size = 2**10, fft_degree = None, time_length = 3,
                        reference_signal = xt)
ni_play.set_output_channels(physical_channel_nums = [2], ao_range = 10)
ni_play.play()

#%% NI playrec
ni_playrec = NIMeasurement(fs = 51200, buffer_size = 2**10, 
                           reference_signal = xt)
ni_playrec.set_output_channels(physical_channel_nums = [1,2], ao_range = 10)
ni_playrec.set_sensor_properties(sensor_type = 'voltage', physical_channel_num = 0, sensitivity = 1, 
                             ai_range = 2)
ni_playrec.set_sensor_properties(sensor_type = 'microphone', physical_channel_num = 3,
                          sensor_current = 4e-3, sensitivity = 54.8, ai_range = 130)

rec_signals = ni_playrec.play_rec()

xtyt = rec_signals.split()
#%% compute h(t)    
ht = pytta.ImpulsiveResponse(excitation = xtyt[0], recording = xtyt[1], samplingRate = ni_playrec.fs, 
                             regularization = True, freq_limits=[freq_min, freq_max])
ht.IR.plot_time();
#%% Ni measurement object

ni_meas = NIMeasurement(reference_sweep=xt, fs = fs, buffer_size = 2**8)
ni_meas.get_system_and_channels()
# ni_meas.set_output_channels(out_channel_to_ni = 3, out_channel_to_amp = 1, ao_range = 10.0)
ni_meas.set_input_channels(in_channel_ref = 3, in_channel_sensor = [0],
                           ai_range = 1, sensor_sens = 50, sensor_current = 2.2e-3)
#ni_meas.initializer()
# First measurement is bad
#ni_meas.play_rec()


#%%
# ni_meas.get_output_task()
# ni_meas.get_input_task()
# time.sleep(2)
# ni_meas.input_task.close()
# ni_meas.output_task.close()
#%% acquire several measurements
n_repeats = 1
# Initialize
ht_mic_xt = np.zeros((2**fft_degree, n_repeats))
ht_mic_lb = np.zeros((2**fft_degree, n_repeats))

# loop through measurements
for jm in range(n_repeats):
    print('Measurement {} of {}'.format(jm+1, n_repeats))
    # Both channels in a single SignalObj
    # xt_ref, yt_mic = ni_meas.play_rec()
    meas_sig_obj = ni_meas.play_rec2()
    meas_sig_obj.split()
    
    # compute h(t)    
    ht_vs_xt = pytta.ImpulsiveResponse(excitation = xt, recording = meas_sig_obj[1], samplingRate = fs, 
                                 regularization = True, freq_limits = [freq_min, freq_max])
    
    ht_ch1_ch2 = pytta.ImpulsiveResponse(excitation = meas_sig_obj[0], recording = meas_sig_obj[1], samplingRate = fs, 
                                 regularization = True, freq_limits = [freq_min, freq_max])
    
    # Feed numpy array
    ht_mic_xt[:, jm] = ht_vs_xt.IR.timeSignal[:,0]
    ht_mic_lb[:, jm] = ht_ch1_ch2.IR.timeSignal[:,0]
    
    # pause
    time.sleep(0.3)

#%% Feed a SignalObj with all measurements
ht_mic_xt_pytta = pytta.classes.SignalObj(signalArray = ht_mic_xt, 
    domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)

ht_mic_lb_pytta = pytta.classes.SignalObj(signalArray = ht_mic_lb, 
    domain='time', freqMin = freq_min, freqMax = freq_max, samplingRate = fs)

#%%
#ht_mic_xt_pytta.plot_time()
ht_mic_lb_pytta.plot_time() #xLim = (0, 0.006)

#%% 
plt.figure(figsize = (8, 4))
plt.plot(ht_mic_lb_pytta.timeVector, 20*np.log10(np.abs(ht_mic_lb)), alpha = 0.8)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
plt.xlim((-0.1, ht_mic_lb_pytta.timeVector[-1]))
#plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()

#%%
plt.figure(figsize = (8, 4))
plt.plot(ht_mic_lb_pytta.timeVector, 20*np.log10(np.abs(ht_mic_lb_pytta.timeSignal)/np.amax(np.abs(ht_mic_lb_pytta.timeSignal))), alpha = 0.8)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
plt.xlim((-0.1, ht_mic_lb_pytta.timeVector[-1]))
plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()
#%%
# path = 'D:/Work/UFSM\Pesquisa/insitu_arrays/experimental_dataset/reptest_eric/loopback_test/'
# suffix = '_fftdegree19_stopmargin1_nreps' + str(n_repeats) + '.hdf5'
# filename_htxt = 'NI34cm_ht_mic_xt' + suffix
# pytta.save(path + filename_htxt, ht_mic_xt_pytta)

# filename_htlb = 'NI34cm_ht_mic_lb' + suffix
# pytta.save(path + filename_htlb, ht_mic_lb_pytta)

# #%%
# med_dict = pytta.load(path + filename_htxt)
# keyslist = list(med_dict.keys())
# ht_mic_xt_pytta = med_dict[keyslist[0]]  


#%%
import nidaqmx as ni
from nidaqmx.constants import WAIT_INFINITELY

def query_devices():
    """Queries all the device information connected to the local system."""
    local = ni.system.System.local()
    for device in local.devices:
        print(f"Device Name: {device.name}, Product Type: {device.product_type}")
        print("Input channels:", [chan.name for chan in device.ai_physical_chans])
        print("Output channels:", [chan.name for chan in device.ao_physical_chans])

def playrec(data, samplerate, input_mapping, output_mapping):
    """Simultaneous playback and recording though NI device.

    Parameters:
    -----------
    data: array_like, shape (nsamples, len(output_mapping))
      Data to be send to output channels.
    samplerate: int
      Samplerate
    input_mapping: list of str
      Input device channels
    output_mapping: list of str
      Output device channels

    Returns
    -------
    ndarray, shape (nsamples, len(input_mapping))
      Recorded data

    """
    devices = ni.system.System.local().devices
    data = np.asarray(data).T
    nsamples = data.shape[1]

    with ni.Task() as read_task, ni.Task() as write_task:
        for i, o in enumerate(output_mapping):
            aochan = write_task.ao_channels.add_ao_voltage_chan(
                o,
                min_val=devices[o].ao_voltage_rngs[0],
                max_val=devices[o].ao_voltage_rngs[1],
            )
            min_data, max_data = np.min(data[i]), np.max(data[i])
            if ((max_data > aochan.ao_max) | (min_data < aochan.ao_min)).any():
                raise ValueError(
                    f"Data range ({min_data:.2f}, {max_data:.2f}) exceeds output range of "
                    f"{o} ({aochan.ao_min:.2f}, {aochan.ao_max:.2f})."
                )
        for i in input_mapping:
            read_task.ai_channels.add_ai_voltage_chan(i)

        for task in (read_task, write_task):
            task.timing.cfg_samp_clk_timing(
                rate=samplerate, source="OnboardClock", samps_per_chan=nsamples
            )

        # trigger write_task as soon as read_task starts
        write_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            read_task.triggers.start_trigger.term
        )
        # squeeze as Task.write expects 1d array for 1 channel
        write_task.write(data.squeeze(), auto_start=False)
        # write_task doesn't start at read_task's start_trigger without this
        write_task.start()
        # do not time out for long inputs
        indata = read_task.read(nsamples, timeout=WAIT_INFINITELY)

    return np.asarray(indata).T



query_devices()
# Prints in this example:
#   Device Name: Dev2, Product Type: USB-4431
#   Input channels: ['Dev2/ai0', 'Dev2/ai1', 'Dev2/ai2', 'Dev2/ai3']
#   Output channels: ['Dev2/ao0']

# excite through one output and record at three inputs
outdata = np.random.normal(size=(5000, 1)) * 0.01
indata = playrec(
    outdata,
    samplerate=51200,
    input_mapping=["cDAQ1Mod1/ai0"],
    output_mapping=["cDAQ1Mod4/ao1"],
)
