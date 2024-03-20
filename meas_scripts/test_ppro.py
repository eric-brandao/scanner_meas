# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:10:55 2022

@author: ericb
"""
#%% 
import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sequential_measurement import ScannerMeasurement
import utils
import pytta
import SSRfunctions

from receivers import Receiver
from sources import Source
#%% Ideally we load the measurement we just did. We will parse for testing
name = 'testing_ppro'
main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash
# arduino_dict = dict()
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 51200, fft_degree = 18, start_stop_margin = [0, 4.5], 
    mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7])

receiver_obj = Receiver()
# receiver_obj.double_planar_array(x_len=0.1,n_x=2,y_len=0.1,n_y=2, zr=0.015, dz=0.03)
receiver_obj.double_rec(z_dist = 0.02)
pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"
meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)

meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 1,
                       freq_max = 25600, n_zeros_pad = 0)
#%% Load Gabriel's measurement
meas_main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/porous_exp_data/double_planar_array_x28y32cm_nx8ny10pts_zr13_dz29mm/Melamine/Measured_data/'
t_bypass_all = np.load(meas_main_folder + 't_bypass.npy')
# Melamine -- measured with double planar array 28 x 32 cm. Also, n_x = 8, n_y = 10, zr = 1,3 cm, dz = 2,9 cm:
yt_mel = np.load(meas_main_folder + 'yt_melamine.npy')
ordened_coord_mel = np.load(meas_main_folder + 'ordened_coord_mel.npy')
meas_obj.receivers.coord = ordened_coord_mel
T_mel = 15.1 # Temperature
yt = yt_mel; ordened_coord = ordened_coord_mel; Temp = T_mel
#%%
plt.figure(figsize=(7,5))
plt.plot(yt[0,0,:], 'k',linewidth = 1, alpha = 0.7)
plt.plot(yt[0,1,:], 'b',linewidth = 1, alpha = 1)

#%% Parse
yt_rec_obj = pytta.classes.SignalObj(signalArray = yt[13,0,:], 
                   domain='time', freqMin = 1, freqMax = 25600, samplingRate = 51200)

repetitions = yt.shape[1]
bar = tqdm(total = yt.shape[0]*yt.shape[1], 
       desc = 'Parsing measured signals to a series of hdf5')
for jrec in range(yt.shape[0]):
    for jmeas in range(repetitions):
        # Pass to pytta
        yt_obj = pytta.classes.SignalObj(signalArray = yt[jrec, jmeas,:], 
              domain='time', freqMin = 1, freqMax = (meas_obj.fs)/2, 
              samplingRate = meas_obj.fs)
        # # ptta saving
        # filename = 'rec' + str(int(jrec)) +\
        #     '_m' + str(int(jmeas)) + '.hdf5'
        # complete_path = meas_obj.main_folder / meas_obj.name / 'measured_signals'
        # pytta.save(str(complete_path / filename), yt_obj)
        bar.update(1)
bar.close()


# utils.parse_meas(yt, ordened_coord, main_folder = meas_main_folder, name = 'melamine',                 
#             fs = 51200, fft_degree = 18, start_stop_margin = [0, 4.5], 
#             mic_sens = None)

#%%
"Temporal Average: ---------------------------------------------------------------------------"
#yt_averaged = []; 
yt_averagedObj = []
bar = tqdm(total = len(yt), desc = 'importing signals')
for i in range(len(yt)):
    point_av = SSRfunctions.temp_average(yt[i])
    p_sumObj = pytta.SignalObj(signalArray=point_av, domain='time', freqMin=1, freqMax=25600, samplingRate=51200)
    #yt_averaged.append(point_av); 
    yt_averagedObj.append(p_sumObj)
    bar.update(1)
bar.close()

#%%
"Processing all Impulsive Responses: ---------------------------------------------------------"
IRs_array = []
bar = tqdm(total = len(yt), desc = 'processing IRs')
for i in range(len(yt)):
    pointIR = pytta.ImpulsiveResponse(excitation=meas_obj.xt, 
          recording=yt_averagedObj[i], samplingRate=51200, 
          regularization=False) 
    IRs_array.append(pointIR)
    bar.update(1)
bar.close()

#%%
"Temporal windows - creation and application -------------------------------------------------"
pts_windowed = []
bar = tqdm(total = len(yt), desc = 'windowing IRs')
for i in range(len(IRs_array)):
    if i == 10:
        pt_win, win = SSRfunctions.IRwindow(t=IRs_array[0].IR.timeVector, 
                pt=IRs_array[i].IR.timeSignal, hss=1.27, d_sample=0.035, 
                d10=0.013, tw1=0.8, tw2=1.5, timelength_w3=0.0094, 
                t_start=0.0002, T=18.0, plot=True)
    else:
        pt_win, win = SSRfunctions.IRwindow(t=IRs_array[0].IR.timeVector,
                pt=IRs_array[i].IR.timeSignal, hss=1.27, d_sample=0.035, 
                d10=0.013, tw1=0.8, tw2=1.5, timelength_w3=0.0094,
                t_start=0.0002, T=18.0, plot=False)
    pts_windowed.append(pt_win)
    bar.update(1)
bar.close()
del(pt_win, win)
    
#%%
plt.figure(figsize=(7,5))
for i in range(50):
    plt.plot(pts_windowed[i], linewidth = 1, alpha = 0.7)
plt.xlim((100, 700))

#%%
plt.figure(figsize=(7,5))
plt.plot(pts_windowed[0], linewidth = 1, alpha = 0.7)
plt.plot(pts_windowed[159], linewidth = 1, alpha = 0.7)
plt.xlim((100, 700))

#%%
def wtf(yt):
    yt_rec_obj = pytta.classes.SignalObj(signalArray = yt, 
                   domain='time', freqMin = 1, freqMax = 25600, samplingRate = 51200)
    return yt_rec_obj