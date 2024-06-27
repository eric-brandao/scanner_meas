# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:43:53 2024

@author: Admin
"""
import matplotlib.pyplot as plt
import numpy as np
from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver
from qterm_estimation import ImpedanceDeductionQterm
from ppro_meas_insitu import InsituMeasurementPostPro

#%% Name of things
name = 'Melamina_13_06_5_FonteProxima' #'PET_grooved_plate' # 'melamine' #
main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/experimental_dataset/MEAS_JoaoP_GNetto/13_06/'

#%% Intantiate post processin object - it will load the meas_obj
ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name)
source = Source([0, 0, 0.3])
ppro_obj.meas_obj.source = source
#%% Compute all IR
ppro_obj.compute_all_ir_load(regularization = True, only_linear_part = True)

#%% Load all IR 
ppro_obj.load_irs()

#%% Plot - check for good health
tlims = (0.01, 0.05)
fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
ppro_obj.plot_ir(ax, idir = 0, normalize = True, xlims = tlims, windowed = False)
ppro_obj.plot_ir(ax, idir = 1, normalize = True, xlims = tlims, windowed = False)
ax.grid()

#%% Set the Adrienne window and apply it on IRs - plot the result
adrienne = ppro_obj.set_adrienne_win(tstart = 18e-3, dt_fadein = 1e-3, t_cutoff = 30e-3, dt_fadeout = 2e-3)
ppro_obj.apply_window()

rec_index = 1
ppro_obj.ir_raw_vs_windowed(idir = rec_index, xlims = (15e-3, 40e-3))
ppro_obj.frf_raw_vs_windowed(idir = rec_index, ylims = (-100, 0))

#%% Reset frequency resolution
ppro_obj.reset_freq_resolution(freq_init = 100, freq_end = 4000, delta_freq = 10)

#%% Estimate absorption and plot
air = AirProperties(c0 = 343.0, rho0 = 1.21,)
controls = AlgControls(c0 = air.c0, freq_vec = ppro_obj.freq_Hw) 

h_pp = ImpedanceDeductionQterm(p_mtx=ppro_obj.Hww_mtx, controls=controls, 
                               receivers=ppro_obj.meas_obj.receivers, 
                               source=ppro_obj.meas_obj.source)
h_pp.pw_pp()
h_pp.pwa_pp()
#h_pp.zq_pp(h_pp.Zs_pwa_pp, tol = 1e-6, max_iter = 40);

plt.figure()
plt.semilogx(h_pp.controls.freq, h_pp.alpha_pw_pp, label = 'PW')
plt.semilogx(h_pp.controls.freq, h_pp.alpha_pwa_pp, label = 'PWA')
#plt.semilogx(h_pp.controls.freq, h_pp.alpha_q_pp, label = 'q-term')
plt.legend()
plt.ylim((-0.4, 1.0))
plt.xlim((100, 4000))
plt.grid()
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'$\alpha$ [-]')
plt.tight_layout();
