# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:36:30 2023

@author: Admin
"""

import sys
sys.path.append('D:/Work/dev/scanner_meas/scanner')
sys.path.append('D:/Work/dev/scanner_meas/process_meas')
import numpy as np
import matplotlib.pyplot as plt
#from sequential_measurement import ScannerMeasurement
from ppro_meas_insitu import InsituMeasurementPostPro
# Receiver class
from receivers import Receiver
from sources import Source
import pytta
#%%
name = 'PET_wool'
main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/exp_data_gabriel_master/raw_data_ppro'# use forward slash
#meas_obj = ScannerMeasurement(main_folder = main_folder, name = name)
#meas_obj.device = 14
#meas_obj.method = 'logarithmic'
#meas_obj.freq_min = 100
#meas_obj.freq_max = 10000
#meas_obj.n_zeros_pad = 0
#meas_obj.load()

#%%
fs = 44100
start_margin = 0.1
stop_margin = 3.5
xt = pytta.generate.sweep(freqMin = 1, freqMax = fs/2, samplingRate= fs,
    startMargin = start_margin, stopMargin = stop_margin, method = 'logarithmic')

#%% 
load_from_folder =  'PET_wool/' #'sonex/' #'PET_grooved_plate/' #'melamine/'#
#freq_vec = np.load(load_from_folder+'f_to_decomp.npy')
rec_coord = np.load('D:/Work/UFSM/Pesquisa/insitu_arrays/exp_data_gabriel_master/' +\
                    load_from_folder+'ordened_coord_to_decomp.npy')
receivers = Receiver()
receivers.coord = rec_coord

source = Source([0, 0, 1.1-0.05])

#%%
ppro_obj = InsituMeasurementPostPro(meas_obj = None, xt = xt, fs = 44100, 
                                    main_folder = main_folder, name = name,
                                    receivers = receivers, 
                                    source = source, repetitions = 2, t_bypass = 13e-3)

#%%
# yt_list = ppro_obj.load_meas_files()

#%%
ht_list = ppro_obj.compute_all_ir_load(regularization = True, only_linear_part = True)

#%%
ppro_obj.load_irs()

#%%
Hw_sm = ppro_obj.moving_avg(idir = 0, nfft = 8192)

#%%
plt.figure()
# plt.semilogx(ppro_obj.freq_Hw, 20*np.log10(np.abs(ppro_obj.Hww_mtx[0,:])))
# plt.semilogx(ppro_obj.freq_Hw, 20*np.log10(np.abs(Hw_sm)))

plt.semilogx(ppro_obj.freq_Hw, np.imag(ppro_obj.Hww_mtx[0,:]))
plt.semilogx(ppro_obj.freq_Hw, np.imag(Hw_sm))
plt.xlim((100, 10000))

#%%
fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
ppro_obj.plot_ir(ax, idir = 0, normalize = True, xlims = (0, 30e-3))

#%%
ht0 = ppro_obj.load_ir_byindex(idir = 0)
ht7 = ppro_obj.load_ir_byindex(idir = 7)
ht11 = ppro_obj.load_ir_byindex(idir = 11)

adrienne = ppro_obj.set_adrienne_win(tstart = 0.0025, dt_fadein = 1e-3, t_cutoff = 13e-3, dt_fadeout = 2e-3,
                     window_size = len(ht0.timeSignal))

#%%
ppro_obj.apply_window()

#%%
ppro_obj.ir_raw_vs_windowed(idir = 0, xlims = (0, 30e-3))
#%%
ppro_obj.plot_all_ir(figsize = (18,12), figformat = (7,9), normalize = True, xlims = (0, 30e-3),
                     windowed = True)

#%%
fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
ppro_obj.plot_frf_mag(ax, idir = 0, xlims = (20, 20000), windowed = True)

#%%
ppro_obj.frf_raw_vs_windowed(idir = 0, ylims = (-100, 0))

#%%
ppro_obj.plot_all_wfrf(figsize = (18,12), figformat = (7,9), xlims = (20, 20000), ylims = (-80, -20))

#%%



ht_windowed = ht0.timeSignal.flatten() * adrienne
Hw_windowed = np.fft.fft(ht_windowed)

plt.figure()
plt.plot(ht0.timeVector, ht0.timeSignal/np.amax(ht0.timeSignal))
plt.plot(ht0.timeVector, ht_windowed/np.amax(ht_windowed))

plt.plot(ht7.timeVector, ht7.timeSignal/np.amax(ht7.timeSignal))
plt.plot(ht11.timeVector, ht11.timeSignal/np.amax(ht11.timeSignal))
plt.plot(ht0.timeVector, adrienne, 'k')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$h(t)/max(h(t))$')
plt.xlim((0.0, 0.10))

plt.figure()
plt.semilogx(ht0.freqVector, 20*np.log10(np.abs(ht0.freqSignal)))
plt.semilogx(ht0.freqVector, 20*np.log10(np.abs(Hw_windowed[:len(ht0.freqSignal)])/len(ht0.freqSignal)))
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$h(t)/max(h(t))$')
plt.xlim((20, 20000))
#%%
# freq inicial, final e resolução
Df = ppro_obj.freq_Hw[1]
Dfn = 5
freq_init = 100
freq_end = 4000
freq_init_idf = np.where(ppro_obj.freq_Hw <= freq_init)[0][-1]
freq_end_idf = np.where(ppro_obj.freq_Hw >= freq_end)[0][0]

# Novo vetor de frequências
Didf = int(Dfn/Df)
freqn = ppro_obj.freq_Hw[freq_init_idf:freq_end_idf+Dfn:Didf]

# novas FRFS
Hf = ppro_obj.Hww_mtx[:,freq_init_idf:freq_end_idf+Dfn:Didf]

# plot
plt.figure(figsize = (8, 4))
plt.semilogx(freqn, 20*np.log10(np.abs(Hf[0,:])))
plt.xlim((freq_init, freq_end))
#plt.ylim((-30, 10))
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$|H(f)|$ [Hz]')
plt.grid()
plt.tight_layout()

#%%
sys.path.append('D:/Work/dev/insitu_sim_python/insitu')
from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from material import PorousAbsorber

# Decomposition
from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M
#%%
ng = 25
a = 0
b = 30
retraction = 0

# Air properties and controls
air = AirProperties(c0 = 343.0, rho0 = 1.21,)
controls = AlgControls(c0 = air.c0, freq_vec = freqn) 

#%%
dcism_array0 = Decomposition_QDT(p_mtx=Hf, controls=controls, receivers=receivers, 
                                 source_coord=source.coord[0], 
    quad_order=ng, a = a, b = b, retraction = retraction, image_source_on = True, regu_par = 'l-curve')
dcism_array0.gauss_legendre_sampling()

dcism_array0.pk_tikhonov(plot_l=False, method='Tikhonov')
#dcism_array0.least_squares_pk()
dcism_array0.zs(Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=[0], avgZs=True);

#%%

plt.figure()
plt.semilogx(dcism_array0.controls.freq, dcism_array0.alpha[0,:], label = 'DCISM (A0)')
plt.legend()
plt.ylim((-0.2, 1.0))
plt.xlim((100, 4000))
plt.grid()
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'$\alpha$ [-]')
plt.tight_layout();