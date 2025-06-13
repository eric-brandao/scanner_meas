# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:54:47 2022

Module to control post processing o material measurements


@author: ericb
"""

# general imports
# import sys
# import os
from pathlib import Path
# import time
from tqdm import tqdm
# import pickle
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import scipy.io as io
from scipy.signal import windows, resample, chirp, find_peaks, find_peaks_cwt

# Pytta imports
import pytta

# # Receiver class
# from receivers import Receiver
# from sources import Source

# utils
import utils
from sequential_measurement import ScannerMeasurement

class InsituMeasurementPostPro():
    """ class to do post processing of insitu measurement
    """
    def __init__(self, main_folder = 'D:', name = 'samplename', t_bypass = 0):               
                 # fs = 51200, meas_obj = None, xt = None, 
                 # receivers = None, source = None,
                 # repetitions = 2, t_bypass = 0):
        
        # load your measurement object without permission to start a new measurement
        self.meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
                                           start_new_measurement = False)
        
        # load all measured information
        self.meas_obj.load()
        # Correct the main_folder variable
        self.meas_obj.main_folder = Path(main_folder)
        
        # if meas_obj is not None:
        #     self.main_folder = meas_obj.main_folder
        #     self.name = meas_obj.name
        # else:
        #     self.main_folder = Path(main_folder)
        #     self.name = name
        # self.receivers = receivers
        # self.source = source
        # self.repetitions = 2
        # self.xt = xt
        # self.fs = fs
        self.t_bypass = t_bypass
        
    def load_meas_byindex(self, idrec = 0, idmed = 0, folder_type = 'measured_signals'):
        """" Load measurement by index of array mic, and return the pytta object
        
        Parameters
        ------------
        idrec : int
            Index of recording to be loaded - array index
        idmed : int
            Index of measurement to be loaded - repetition index       
        """
        # rec0_m0
        filename = 'rec' + str(int(idrec)) + '_m' + str(int(idmed)) + '.hdf5'
        complete_path = self.meas_obj.main_folder / self.meas_obj.name / folder_type
        med_dict = pytta.load(str(complete_path / filename))
        keyslist = list(med_dict.keys())
        yt = med_dict[keyslist[0]]
        return yt
        
    def load_allmeas_files(self, id_med = 0):
        """Load all measurement files
        """
        yt_list = []
        for jrec in range(self.meas_obj.receivers.coord.shape[0]):
            y_rep_list = []
            for jmeas in range(self.meas_obj.repetitions):
                yt = self.load_meas_byindex(idrec = jrec, idmed = jmeas)
                y_rep_list.append(yt)
            yt_list.append(y_rep_list)
        return yt_list
    
    # def ir(self, yt, regularization = True):
    #     """ Computes the impulse response of a given output
        
    #     Parameters
    #     ----------
    #     yt : pytta object
    #         output signal
    #     """
    #     ht = pytta.ImpulsiveResponse(excitation = self.meas_obj.xt, 
    #          recording = yt, samplingRate = self.meas_obj.xt.samplingRate, regularization = regularization,
    #          method = 'linear')
        
    #     return ht
    
    def mean_ir(self, ht_rep_list, only_linear_part = True):
        """Computes mean IR as in Pytta"""
        ht_stack = np.zeros((ht_rep_list[0].timeSignal.shape[0], self.meas_obj.repetitions),
                            dtype=ht_rep_list[0].timeSignal.dtype)
        
        for jmeas in range(self.meas_obj.repetitions):
            ht_stack[:, jmeas] = ht_rep_list[jmeas].timeSignal.flatten()
        
        ht_mean = np.mean(ht_stack, axis = 1, dtype=ht_rep_list[0].timeSignal.dtype)
        
        if only_linear_part:
            ht_mean = ht_mean[:int(len(ht_mean)/2)]
        
        ht_mean_pytta = pytta.SignalObj(signalArray=ht_mean,
                          lengthDomain='time', samplingRate=ht_rep_list[0].samplingRate)
        return ht_mean_pytta
    
    def compute_all_ir(self, yt_list, regularization = True, 
                       only_linear_part = True, bar_leave = True):
        """Compute all Impulse responses
        """
        ht_list = []
        bar = tqdm(total = self.meas_obj.receivers.coord.shape[0], leave = bar_leave,
                desc = 'Processing IRs')
        # For each receiver compute repeated ht
        for jrec in range(self.meas_obj.receivers.coord.shape[0]):
            ht_rep_list = []
            # loop through repetitions
            for jmeas in range(self.meas_obj.repetitions):
                ht = self.ir(yt_list[jrec][jmeas], regularization = regularization)
                ht_rep_list.append(ht.IR)
            # take mean IR
            ht_mean_pytta = self.mean_ir(ht_rep_list, only_linear_part = only_linear_part)
            ht_list.append(ht_mean_pytta)
            bar.update(1)
        bar.close()
        return ht_list
    
    def compute_all_ir_load(self, regularization = True,  deconv_with_rec = True,
                            freq_limits = None, only_linear_part = True, 
                            reverse_phase = False):
        
       """Compute all Impulse responses while loading measurement files. Saves memory
       """
       
       # For each receiver compute repeated ht
       for jrec in range(self.meas_obj.receivers.coord.shape[0]):
           print('Loading and computing IR for Rec {}'.format(jrec))
           ht_rep_list = []
           # loop through repetitions
           for jmeas in range(self.meas_obj.repetitions):
               # Load measurement yt files
               filename = 'rec' + str(int(jrec)) +\
                       '_m' + str(int(jmeas)) + '.hdf5'
               complete_path = self.meas_obj.main_folder / self.meas_obj.name / 'measured_signals'
               med_dict = pytta.load(str(complete_path / filename))
               keyslist = list(med_dict.keys())
               yt = med_dict[keyslist[0]]               
               
               # Compute ht
               ht = self.meas_obj.ir(yt, regularization = regularization,
                                     deconv_with_rec =  deconv_with_rec,
                                     freq_limits = freq_limits,
                                     reverse_phase = reverse_phase)
               ht_rep_list.append(ht.IR)
           # take mean IR
           ht_mean_pytta = self.mean_ir(ht_rep_list, only_linear_part = only_linear_part)
           
           # Discount the bypass
           # ht_mean_pytta.crop(float(self.t_bypass), float(ht_mean_pytta.timeVector[-1]))
           
           # ptta saving
           filename = 'ht' + str(int(jrec)) + '.hdf5'
           complete_path = self.meas_obj.main_folder / self.meas_obj.name / 'impulse_responses'
           pytta.save(str(complete_path / filename), ht_mean_pytta)
           
       
    def load_ir_byindex(self, idir = 0):
        """" Load IR by index of array mic, and return the pytta object"""
        
        filename = 'ht' + str(int(idir)) + '.hdf5'
        complete_path = self.meas_obj.main_folder / self.meas_obj.name / 'impulse_responses'
        med_dict = pytta.load(str(complete_path / filename))
        keyslist = list(med_dict.keys())
        ht = med_dict[keyslist[0]]
        return ht
    
    def load_irs(self,):
        """ Load all IRs to a matrix
        """
        # load 0 case
        ht = self.load_ir_byindex(0)
        
        # initialize
        self.ht_mtx = np.zeros((self.meas_obj.receivers.coord.shape[0], len(ht.timeSignal)))
        self.ht_mtx[0, :] = ht.timeSignal.flatten()
        # For each receiver compute repeated ht
        for jrec in range(1, self.meas_obj.receivers.coord.shape[0]):
            ht = self.load_ir_byindex(jrec)
            self.ht_mtx[jrec, :] = ht.timeSignal.flatten()
        self.time_ht = ht.timeVector.flatten()
        self.ht_length = len(self.time_ht)
        # print("ht matrix has {:.2f} MB".format(ht_mtx.nbytes/(1024*1024)))
        
    def load_irs2(self,):
        """ Load all IRs to a matrix
        """
        # load 0 case
        ht = self.load_ir_byindex(0)
        
        # initialize
        self.ht_mtx = np.zeros((self.meas_obj.receivers.coord.shape[0], len(ht.IR.timeSignal)))
        self.ht_mtx[0, :] = ht.IR.timeSignal.flatten()
        # For each receiver compute repeated ht
        for jrec in range(1, self.meas_obj.receivers.coord.shape[0]):
            ht = self.load_ir_byindex(jrec)
            self.ht_mtx[jrec, :] = ht.IR.timeSignal.flatten()
        self.time_ht = ht.IR.timeVector.flatten()
        self.ht_length = len(self.time_ht)
        # print("ht matrix has {:.2f} MB".format(ht_mtx.nbytes/(1024*1024)))
    
    def move_ir(self, idir = 0, c0 = 340, 
                source_coord = [0, 0, 1], receiver_coord = [0, 0, 0.01],
                plot_ir = False, xlims = (0, 50e-3)):
        """ Move the IR to physical starting point in time. 
        
        It uses an estimate of sound speed and sensor location to compute the onset of the IR.
        """
        
        
        # compute the physical onset time of IR
        euclidian_distance = np.linalg.norm(source_coord - receiver_coord)
        time_onset = euclidian_distance/c0
        
        # Find main peak
        ht_peak_fun = self.ht_mtx[idir,:]
        # ht_peak_fun = self.ht_mtx[idir,:]/np.amax(self.ht_mtx[idir,:])
        # ht_peak_fun = self.ht_mtx[idir,:]/np.amax(np.abs(self.ht_mtx[idir,:]))
        # ht_peak_fun = np.abs(self.ht_mtx[idir,:])/np.amax(np.abs(self.ht_mtx[idir,:]))
        # ht_peak_fun = np.abs(self.ht_mtx[idir,:]/np.amax(np.abs(self.ht_mtx[idir,:])))

        # peaks_id = find_peaks(ht_peak_fun, height = 0.9)
        peaks_id = np.where(ht_peak_fun == np.amax(ht_peak_fun))
        
        # move whole thing to id 0
        # self.ht_mtx[idir,:] = np.roll(self.ht_mtx[idir,:], shift = -peaks_id[0][0])
        
        # peaks_id = find_peaks_cwt(np.abs(self.ht_mtx[idir,:]/np.amax(np.abs(self.ht_mtx[idir,:]))),
        #                           np.ones(len(self.ht_mtx[idir,:])))
        # print(peaks_id)
        time_of_peak = peaks_id[0][0]/self.meas_obj.fs
        
        # find out how many samples to move
        delta_t = time_of_peak - time_onset
        # delta_t = np.round(time_of_peak, 6) - np.round(time_onset,6)
        
        
        print("time_onset = {}, delta_t = {}, Ts = {}".format(time_onset, delta_t, 1/self.meas_obj.fs))
        # n_samples_to_move = int(np.rint(delta_t * self.meas_obj.fs))
        # n_samples_to_move = int(delta_t * self.meas_obj.fs)
        n_samples_to_move = int(np.ceil(delta_t * self.meas_obj.fs))
        # n_samples_to_move = int(np.ceil(time_onset * self.meas_obj.fs))
        print("samples moved: {}".format(n_samples_to_move))
        # n_samples_to_move = int(np.floor(delta_t * self.meas_obj.fs))
        # print(n_samples_to_move)
        self.ht_mtx[idir,:] = np.roll(self.ht_mtx[idir,:], shift = -n_samples_to_move)
        # move to onset
        # self.ht_mtx[idir,:] = np.roll(self.ht_mtx[idir,:], shift = n_samples_to_move)
        # print("Delta t = {} s, Num of samples to move = {}".format(delta_t, n_samples_to_move))
        
        # plot
        if plot_ir:
            # normalize
            ht = self.ht_mtx[idir,:]/np.amax(self.ht_mtx[idir,:])
            # plot
            plt.figure()
            plt.plot(self.time_ht, ht, label = "Rec #{}".format(idir))
            plt.xlim(xlims)
            plt.axvline(x = time_onset, color = 'grey', linestyle = '--')
            plt.axvline(x = time_of_peak, color = 'k', linestyle = '--')
            plt.grid()
            plt.xlabel("Time (s)")
            plt.ylabel("Time (s)")
            plt.tight_layout()
        
    def move_all_ir(self, c0 = 340):
        """ Move all IRs
        """
        # For each receiver compute repeated ht
        for jrec in range(self.meas_obj.receivers.coord.shape[0]):
            self.move_ir(idir = jrec, c0 = c0, 
                        source_coord = self.meas_obj.source.coord, 
                        receiver_coord = self.meas_obj.receivers.coord[jrec,:])
    
    def set_adrienne_win(self, tstart = 0, dt_fadein = 0.5e-3, t_cutoff = 15e-3, dt_fadeout = 1e-3):
        """ set the Adrienne window
        
        Parameters:
        -------------------------
            t_start : float
               Instance when the window's cte part starts
            t_cutoff : float
               Instance when the window's cte part stops
            dt_fadein : float
               window's fade in duration
            dt_fadeout : float
               window's fade out duration
            window_size : int
               window's number of samples (same as IR) 
        """
        # initiallize
        self.adrienne_win = np.zeros(self.ht_length)
        
        # blackman-harris for fade in
        bh_fadein = windows.blackmanharris(int(2*dt_fadein*self.meas_obj.fs))
        bh_fadein = bh_fadein[:int(len(bh_fadein)/2)]
        
        # blackman-harris for fade out
        bh_fadeout = windows.blackmanharris(int(2*dt_fadeout*self.meas_obj.fs))
        bh_fadeout = bh_fadeout[int(len(bh_fadeout)/2):]
        
        # Adrienne win during fade in
        start_sample = int((np.abs(tstart-dt_fadein))*self.meas_obj.fs)
        self.adrienne_win[start_sample:start_sample+len(bh_fadein)] = bh_fadein
        
        # Adrienne win cte
        stop_sample = int(t_cutoff*self.meas_obj.fs)
        self.adrienne_win[start_sample+len(bh_fadein):stop_sample] = 1
        
        # Adrienne win during fade out
        self.adrienne_win[stop_sample:stop_sample+len(bh_fadeout)] = bh_fadeout
        
        #plt.plot(adrienne_win)
        return self.adrienne_win
    
    def apply_window(self,):
        """ Apply window to impulse responses"""
        self.htw_mtx = self.ht_mtx * self.adrienne_win
        # FFT
        nfft = self.htw_mtx.shape[1]
        if (nfft % 2) == 0:
            self.nfft_half = int(nfft/2)
        else:
            self.nfft_half = int((nfft+1)/2)           
        self.freq_Hw = np.linspace(0, (nfft-1)*self.meas_obj.fs/nfft, nfft)[:self.nfft_half]
        self.Hww_mtx = np.fft.fft(self.htw_mtx, axis = 1)[:,:self.nfft_half]
        
    def compute_spk(self,):
        """ Computes the spectrum on the time signal matrix
        """
        # FFT
        nfft = self.ht_mtx.shape[1]
        if (nfft % 2) == 0:
            self.nfft_half = int(nfft/2)
        else:
            self.nfft_half = int((nfft+1)/2)
            
        self.freq_Hw = np.linspace(0, (nfft-1)*self.meas_obj.fs/nfft, nfft)[:self.nfft_half]
        self.Hw_mtx = np.fft.fft(self.ht_mtx, axis = 1)[:,:self.nfft_half]
    
    def pcc_magspk(self, yt_list, ref_ch = 1):
        """ Computes the PCC between the magnitude of a recording and the ref. sweep
        
        Performs the calculations for all recordings in yt_list
        
        Parameters
        ----------
        yt_list : list
            list of all singal objects (pytta)
        ref_ch : int
            Reference channel to compute the PCC
        """
        num_recs = len(yt_list)
        self.pcc_spk = np.zeros((num_recs, self.meas_obj.repetitions))
        bar = tqdm(total = num_recs*self.meas_obj.repetitions,
            desc = 'Computing PCC (SPK) for all signals')
        for jrec in range(num_recs):
            for jrep in range(self.meas_obj.repetitions):
                self.pcc_spk[jrec, jrep] = self.meas_obj.pcc_magspk(yt_list[jrec][jrep],
                                                                    ref_ch = ref_ch)
                bar.update(1)
        bar.close()
    
    def flag_measurements(self, min_pcc = 0.99, spk_pcc = True):
        """ Flags a given measurement
        
        If the computed PCC is lower than min_pcc, the measurement will be flagged as bad.
        
        Parameters
        ----------
        min_pcc : float
            Minimum value of PCC for which the measurement is considered good.
        spk_pcc : bool
            PCC via spectral mode (default is True).
        """
        if spk_pcc:
            pcc_mtx = self.pcc_spk
        else:
            pcc_mtx = self.pcc_spk # The same now. Later we compute it in time dommain
        # Find problematic measurements
        self.problematic_measurements = np.where(np.any(pcc_mtx < min_pcc, axis=1))[0]
    
    def reset_freq_resolution(self, freq_init = 100, freq_end = 4000, delta_freq = 5):
        """ If you don't want all your frequencies, use this to generate a new
        self.Hwww_mtx
        """
        # initial freq resolution
        delta_freq_original = self.freq_Hw[1]
        # indexes of frequencies
        freq_init_idf = np.where(self.freq_Hw <= freq_init)[0][-1]
        freq_end_idf = np.where(self.freq_Hw >= freq_end)[0][0]
        
        # new freq vector
        Didf = int(delta_freq/delta_freq_original)
        self.freq_Hw = self.freq_Hw[freq_init_idf:freq_end_idf+delta_freq:Didf]
        
        # new FRF's
        self.Hww_mtx = self.Hww_mtx[:,freq_init_idf:freq_end_idf+delta_freq:Didf]
        
    def moving_avg(self, idir = 0, nfft = 8192):
        """ Computes moving average on spectrum
        """
        G=1-0.4/(1.5e4)*self.freq_Hw
        Hw_sm = np.zeros(len(self.Hww_mtx[idir,:]), dtype = complex)
        for a in np.arange(0, len(self.freq_Hw)):
            b = np.round([a - G[a] * a * (nfft-1) / self.fs, 
                          a + G[a] * a * (nfft-1) / self.fs])
            
            try:
                mag = np.mean(np.abs(self.Hww_mtx[idir, int(b[0]):int(b[1])]))
                phase = np.mean(np.angle(self.Hww_mtx[idir, int(b[0]):int(b[1])]))
                Hw_sm[a] = mag*np.exp(1j*phase)
                # Hw_sm[a] = np.mean(self.Hww_mtx[idir, int(b[0]):int(b[1])])
            except:
                Hw_sm[a] = self.Hww_mtx[a];
        return Hw_sm

    def plot_signal(self, ax = None, xdata = None, ydata = None, 
                    data_label = 'any', xlabel = 'Time [s]', ylabel = 'Amplitude [-]',
                    xlims = None, ylims = None, xlog = False, alpha = 0.7,
                    linestyle = '-'):
        
        if ax is None: # create a general axis is ax is None
            fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        
        if xlog:
            ax.semilogx(xdata, ydata, label = data_label, alpha = alpha,
                        linestyle = linestyle)
        else:
            ax.plot(xdata, ydata, label = data_label, alpha = alpha,
                    linestyle = linestyle)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(linestyle = '--')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        return ax
    
    # def fetch_sigobj_from_list(yt_list, idrec = 0, idmed = 0):
    #     """ Fetch your signal object
    #     """
    #     return yt_list[idrec][idmed]
        
    def plot_meas_time(self, yt_list, idrec = 0, idmed = 0, xlims = None, ylims = None):
        """ plot single signal in time domain - all channels
        
        Parameters 
        ---------------
        yt : list
            list of all singal objects (pytta)
        idrec : int
            index of recording (receiver)
        idmed : int
            index of measurement (repetition)
        xlims : tuple
            min and max values of your x-axis (limit view)
        ylims : tuple
            min and max values of your y-axis (limit view)
        """
        yt = yt_list[idrec][idmed]
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        for jch in range(yt.timeSignal.shape[1]):
            ax = self.plot_signal(ax = ax, xdata = yt.timeVector, 
                             ydata = yt.timeSignal[:, jch],
                             xlims = xlims, ylims = ylims,
                             xlabel = 'Time [s]', ylabel = r'$y(t)$ [-]',
                             data_label = "Rec. # {}, Rep. {}, Ch. {}".format(idrec, idmed, jch))
            ax.legend()
    
    def plot_meas_spk(self, yt_list, idrec = 0, idmed = 0, 
                      xlims = (20, 20000), ylims = None):
        """ plot single signal in time domain - all channels
        
        Parameters 
        ---------------
        yt : list
            list of all singal objects (pytta)
        idrec : int
            index of recording (receiver)
        idmed : int
            index of measurement (repetition)
        xlims : tuple
            min and max values of your x-axis (limit view)
        ylims : tuple
            min and max values of your y-axis (limit view)
        """
        yt = yt_list[idrec][idmed]
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        for jch in range(yt.timeSignal.shape[1]):
            ax = self.plot_signal(ax = ax, xdata = yt.freqVector, 
                             ydata = 20*np.log10(np.abs(yt.freqSignal[:, jch])),
                             xlims = xlims, ylims = ylims,
                             xlabel = 'Frequency [Hz]', ylabel = r'$|H(f)|$ [-]',
                             data_label = "Rec. # {}, Rep. {}, Ch. {}".format(idrec, idmed, jch),
                             xlog = True)
            ax.legend()
    
    def plot_ir(self, idir = 0, xlims = (0, 20e-3), 
                normalize = True, windowed = False):
        """ plot single Impulse response
        
        Parameters 
        ---------------
        idir : int
            index of the impulse response to plot
        xlims : tuple
            min and max values of your x-axis (limit view)
        normalized : bool
            whether to normalize or not the IR
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        if windowed:
            ht = self.htw_mtx[idir, :]
        else:
            ht = self.ht_mtx[idir, :]
        
        if normalize:
            ht = ht/np.amax(ht)
            
        ax = self.plot_signal(xdata = self.time_ht, ydata = ht, 
                         data_label = "Rec #{}".format(idir), 
                         xlabel = 'Time [s]', ylabel = r'$h(t)$ [-]',
                         xlims = xlims, xlog = False, alpha = 0.7)
        ax.legend()
        
    def ir_raw_vs_windowed(self, idir = 0, xlims = (0, 20e-3),
                           normalize = True):
        """ Compare same IR before and after windowing
        
        Parameters 
        ---------------
        idir : int
            index of the impulse response to plot
        xlims : tuple
            min and max values of your x-axis (limit view)
        normalized : bool
            whether to normalize or not the IR
        """
        if normalize:
            ht = self.ht_mtx[idir,:]/np.amax(self.ht_mtx[idir,:])
            htw = self.htw_mtx[idir,:]/np.amax(self.htw_mtx[idir,:])
        
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        
        ax = self.plot_signal(ax = ax, xdata = self.time_ht, ydata = ht, 
                          data_label = "Rec #{} - raw".format(idir), 
                          xlabel = 'Time [s]', ylabel = r'$h(t)$ [-]',
                          xlims = xlims, xlog = False, alpha = 1.0)
        
        ax = self.plot_signal(ax = ax, xdata = self.time_ht, ydata = htw, 
                          data_label = "Rec #{} - windowed".format(idir), 
                          xlabel = 'Time [s]', ylabel = r'$h(t)$ [-]',
                          xlims = xlims, xlog = False, alpha = 0.7,
                          linestyle = '--')
        
        ax.plot(self.time_ht, self.adrienne_win, '--k', alpha = 0.7, 
                label = 'window')
        
        # ax = self.plot_signal(ax = ax, xdata = self.time_ht, ydata = self.adrienne_win, 
        #                   data_label = "window".format(idir), 
        #                   xlabel = 'Time [s]', ylabel = r'$h(t)$ [-]',
        #                   xlims = xlims, xlog = False, alpha = 0.7,
        #                   linestyle = '--')
        ax.legend()
        
    def plot_frf_mag(self, idir = 0, xlims = None, ylims = None,
                windowed = False):
        """ plot single FRF magnitude
        
        Parameters 
        ---------------
        idir : int
            index of the impulse response to plot
        xlims : tuple
            min and max values of your x-axis (limit view)
        ylims : tuple
            min and max values of your y-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        if windowed:
            Hw = self.Hww_mtx[idir, :]
        else:
            Hw = self.Hw_mtx[idir, :]# np.fft.fft(self.ht_mtx[idir,:])[:self.nfft_half]
        
        ax = self.plot_signal(xdata = self.freq_Hw, ydata = 20*np.log10(np.abs(Hw)), 
                         data_label = "Rec #{}".format(idir), 
                         xlabel = 'Frequency [Hz]', ylabel = r'$|H(f)|$ [-]',
                         xlims = xlims, ylims = ylims, xlog = True, alpha = 0.7)
        ax.legend()
        
    def frf_raw_vs_windowed(self, idir = 0, xlims = None, ylims = None):
        """ Compare Raw vs. Windowed FRF magnitude
        
        Parameters 
        ---------------
        idir : int
            index of the impulse response to plot
        xlims : tuple
            min and max values of your x-axis (limit view)
        ylims : tuple
            min and max values of your y-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        Hw_raw = self.Hw_mtx[idir, :]
        Hw_win = self.Hww_mtx[idir, :]
                
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax = self.plot_signal(ax = ax, 
                              xdata = self.freq_Hw, ydata = 20*np.log10(np.abs(Hw_raw)),
                              data_label = "Rec #{} - raw".format(idir),
                              xlabel = 'Frequency [Hz]', ylabel = r'$|H(f)|$ [-]',
                              xlims = xlims, ylims = ylims, xlog = True, alpha = 1.0)
        
        ax = self.plot_signal(ax = ax, 
                              xdata = self.freq_Hw, ydata = 20*np.log10(np.abs(Hw_win)),
                              data_label = "Rec #{} - windowed".format(idir),
                              xlabel = 'Frequency [Hz]', ylabel = r'$|H(f)|$ [-]',
                              xlims = xlims, ylims = ylims, xlog = True, alpha = 1.0)
        ax.legend()
             
    def num_curves_per_axis(self, figformat):
        """ get number of curves per axes
        """
        num_of_axis = figformat[0]*figformat[1]
        num_of_cur_axis = int(self.meas_obj.receivers.coord.shape[0]/num_of_axis)
        num_of_remaining_cur = self.meas_obj.receivers.coord.shape[0] - num_of_axis*num_of_cur_axis
        return num_of_cur_axis, num_of_remaining_cur
            
    def plot_all_ir(self, figsize = (20, 10), figformat = (6,8),
                    xlims = (0, 20e-3), windowed = False):
        """ plot all irs (normalized)
        
        Parameters 
        ---------------
        figsize : tuple
            size of the final figure
        figformat : tuple
            number of rows and columns of the figure
        xlims : tuple
            min and max values of your x-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        # Number of curves per axis
        num_of_cur_axis, num_of_remaining_cur = self.num_curves_per_axis(figformat = figformat)
        # choose windowed or not
        if windowed:
            ht = self.htw_mtx
        else:
            ht = self.ht_mtx
        # axis
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True, squeeze=False)
        counter = 0
        ax_counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                if ax_counter < num_of_remaining_cur:
                    num_of_curv_2plot = num_of_cur_axis + 1
                else:
                    num_of_curv_2plot = num_of_cur_axis
                starting_curv = counter + 1
                ending_curv = starting_curv + num_of_curv_2plot - 1
                for curv in range(num_of_curv_2plot):
                    ht_plt = ht[counter,:]/np.amax(ht[counter,:])
                    ax[row, col] = self.plot_signal(ax = ax[row, col], 
                                                    xdata = self.time_ht, 
                                                    ydata = ht_plt, 
                                                    xlabel = 'Time [s]', 
                                                    ylabel = r'$h(t)$ [-]',
                                                    xlims = xlims, 
                                                    xlog = False, alpha = 0.7)
                    
                    ax[row, col].set_title("# {}-{}".format(starting_curv, ending_curv),
                                           loc = 'right')
                    counter += 1
                ax_counter += 1
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                
                ax[figformat[0]-1, col].set_xlabel("Time [s]")
            ax[row, 0].set_ylabel(r'$h(t)$ [-]')
        plt.suptitle("Amplitude (IR) / Windowed: {}".format(str(windowed)))
        plt.tight_layout()      
    
    def plot_all_frf(self, figsize = (20, 10), figformat = (6,8),
                    xlims = (20, 20000), ylims = None, windowed = False):
        """ plot all FRF's
        
        Parameters 
        ---------------
        figsize : tuple
            size of the final figure
        figformat : tuple
            number of rows and columns of the figure
        xlims : tuple
            min and max values of your x-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        # Number of curves per axis
        num_of_cur_axis, num_of_remaining_cur = self.num_curves_per_axis(figformat = figformat)
        # choose windowed or not
        if windowed:
            Hw = self.Hww_mtx
        else:
            Hw = self.Hw_mtx
        # axes
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True, squeeze=False)
        counter = 0
        ax_counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                if ax_counter < num_of_remaining_cur:
                    num_of_curv_2plot = num_of_cur_axis + 1
                else:
                    num_of_curv_2plot = num_of_cur_axis
                starting_curv = counter + 1
                ending_curv = starting_curv + num_of_curv_2plot - 1
                for curv in range(num_of_curv_2plot):
                    Hw_plt = 20*np.log10(np.abs(Hw[counter,:]))
                    ax[row, col] = self.plot_signal(ax = ax[row, col], 
                                                    xdata = self.freq_Hw, 
                                                    ydata = Hw_plt, 
                                                    xlabel = 'Frequency [Hz]', 
                                                    ylabel = r'$|H(f)|$ [dB]',
                                                    xlims = xlims,
                                                    ylims = ylims,
                                                    xlog = True, alpha = 0.7)
                    ax[row, col].set_title("# {}-{}".format(starting_curv, ending_curv),
                                           loc = 'right')
                    counter += 1
                ax_counter += 1
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                ax[figformat[0]-1, col].set_xlabel("Frequency [Hz]")
            ax[row, 0].set_ylabel(r"$|H(f)|$ [dB]")
        plt.suptitle("Magnitude (FRF) / Windowed: {}".format(str(windowed)))
        plt.tight_layout()
        
    def plot_all_meas_time(self, yt_list, ch = 0, idmed = 0, figsize = (20, 10), figformat = (6,8),
                    xlims = (20, 20000), ylims = None):
        """ plot all FRF's
        
        Parameters 
        ---------------
        yt_list : list
            list of all singal objects (pytta)
        ch : int
            Which channel to plot
        idmed : int
            index of measurement (repetition) 
        figsize : tuple
            size of the final figure
        figformat : tuple
            number of rows and columns of the figure
        xlims : tuple
            min and max values of your x-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        # Number of curves per axis
        num_of_cur_axis, num_of_remaining_cur = self.num_curves_per_axis(figformat = figformat)
        time = yt_list[0][0].timeVector
        # axes
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True, squeeze=False)
        counter = 0
        ax_counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                if ax_counter < num_of_remaining_cur:
                    num_of_curv_2plot = num_of_cur_axis + 1
                else:
                    num_of_curv_2plot = num_of_cur_axis
                starting_curv = counter + 1
                ending_curv = starting_curv + num_of_curv_2plot - 1
                for curv in range(num_of_curv_2plot):
                    yt_plt = yt_list[counter][idmed].timeSignal[:,ch]
                    ax[row, col] = self.plot_signal(ax = ax[row, col], 
                                                    xdata = time, 
                                                    ydata = yt_plt, 
                                                    xlabel = 'Time [s]', 
                                                    ylabel = r'$y(t)$ [-]',
                                                    xlims = xlims,
                                                    ylims = ylims,
                                                    xlog = False, alpha = 0.7)
                    ax[row, col].set_title("# {}-{}".format(starting_curv, ending_curv),
                                           loc = 'right')
                    counter += 1
                ax_counter += 1
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                ax[figformat[0]-1, col].set_xlabel("Time [s]")
            ax[row, 0].set_ylabel(r'$y(t)$ [-]')
        plt.suptitle("Amplitude (time) of Ch. {}, Rep. {}".format(ch, idmed))
        plt.tight_layout()
        
    def plot_all_meas_spk(self, yt_list, ch = 0, idmed = 0, figsize = (20, 10), figformat = (6,8),
                    xlims = (20, 20000), ylims = None):
        """ plot all FRF's
        
        Parameters 
        ---------------
        yt : list
            list of all singal objects (pytta)
        ch : int
            Which channel to plot
        idmed : int
            index of measurement (repetition) 
        figsize : tuple
            size of the final figure
        figformat : tuple
            number of rows and columns of the figure
        xlims : tuple
            min and max values of your x-axis (limit view)
        windowed : bool
            whether to plot windowed or non-windowed IR (if already computed)
        """
        # Number of curves per axis
        num_of_cur_axis, num_of_remaining_cur = self.num_curves_per_axis(figformat = figformat)
        freq = yt_list[0][0].freqVector
        # axes
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True, squeeze=False)
        counter = 0
        ax_counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                if ax_counter < num_of_remaining_cur:
                    num_of_curv_2plot = num_of_cur_axis + 1
                else:
                    num_of_curv_2plot = num_of_cur_axis
                starting_curv = counter + 1
                ending_curv = starting_curv + num_of_curv_2plot - 1
                for curv in range(num_of_curv_2plot):
                    yt_spk = yt_list[counter][idmed].freqSignal[:,ch]
                    Yw_plt = 20*np.log10(np.abs(yt_spk))
                    ax[row, col] = self.plot_signal(ax = ax[row, col], 
                                                    xdata = freq, 
                                                    ydata = Yw_plt, 
                                                    xlabel = 'Frequency [Hz]', 
                                                    ylabel = r'$|Y(f)|$ [dB]',
                                                    xlims = xlims,
                                                    ylims = ylims,
                                                    xlog = True, alpha = 0.7)
                    ax[row, col].set_title("# {}-{}".format(starting_curv, ending_curv),
                                           loc = 'right')
                    counter += 1
                ax_counter += 1
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                ax[figformat[0]-1, col].set_xlabel("Frequency [Hz]")
            ax[row, 0].set_ylabel(r"$|Y(f)|$ [dB]")
        plt.suptitle("Magnitude (Spk) of Ch. {}, Rep. {}".format(ch, idmed))
        plt.tight_layout()
        
    def save(self, filename = 'qdt', path = ''):
        """ To save the decomposition object as pickle
        """
        temp_dict =  self.__dict__   
        if hasattr(self, 'xt'):
            del temp_dict['xt']
        utils.save(self, filename = filename, path = path)

    def load(self, filename = 'qdt', path = ''):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.
        """
        utils.load(self, filename = filename, path = path)