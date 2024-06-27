# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:54:47 2022

Module to control post processing o material measurements


@author: ericb
"""

# general imports
import sys
import os
from pathlib import Path
import time
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as io
from scipy.signal import windows, resample, chirp

# Pytta imports
import pytta


# Receiver class
from receivers import Receiver
from sources import Source

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
        
    def load_meas_files(self,):
        """Load all measurement files
        """
        yt_list = []
        for jrec in range(self.meas_obj.receivers.coord.shape[0]):
            y_rep_list = []
            for jmeas in range(self.meas_obj.repetitions):
                filename = 'rec' + str(int(jrec)) +\
                        '_m' + str(int(jmeas)) + '.hdf5'
                complete_path = self.meas_obj.main_folder / self.meas_obj.name / 'measured_signals'
                med_dict = pytta.load(str(complete_path / filename))
                keyslist = list(med_dict.keys())
                y_rep_list.append(med_dict[keyslist[0]])
            yt_list.append(y_rep_list)
            
        return yt_list
    
    def ir(self, yt, regularization = True):
        """ Computes the impulse response of a given output
        
        Parameters
        ----------
        yt : pytta object
            output signal
        """
        ht = pytta.ImpulsiveResponse(excitation = self.meas_obj.xt, 
             recording = yt, samplingRate = self.meas_obj.xt.samplingRate, regularization = regularization)
        
        return ht
    
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
    
    def compute_all_ir_load(self, regularization = True, 
                       only_linear_part = True, bar_leave = True):
        
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
               ht = self.ir(yt, regularization = regularization)
               ht_rep_list.append(ht.IR)
           # take mean IR
           ht_mean_pytta = self.mean_ir(ht_rep_list, only_linear_part = only_linear_part)
           
           # Discount the bypass
           ht_mean_pytta.crop(float(self.t_bypass), float(ht_mean_pytta.timeVector[-1]))
           
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
        self.adrienne_win[int((tstart-dt_fadein)*self.meas_obj.fs):int((tstart-dt_fadein)*self.meas_obj.fs)+len(bh_fadein)] = bh_fadein
        
        # Adrienne win cte
        self.adrienne_win[int((tstart-dt_fadein)*self.meas_obj.fs)+len(bh_fadein):int(t_cutoff*self.meas_obj.fs)] = 1
        
        # Adrienne win during fade out
        self.adrienne_win[int(t_cutoff*self.meas_obj.fs):int((t_cutoff)*self.meas_obj.fs)+len(bh_fadeout)] = bh_fadeout
        
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
        self.freq_Hw = np.linspace(0, (nfft-1)*self.fs/nfft, nfft)[:self.nfft_half]
        self.Hww_mtx = np.fft.fft(self.ht_mtx, axis = 1)[:,:self.nfft_half]
        
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

    
    def plot_ir(self, ax, idir = 0, normalize = True, xlims = (0, 50e-3),
                windowed = False):
        """ plot an axis with ht
        """
        if windowed:
            ht = self.htw_mtx[idir, :]
        else:
            ht = self.ht_mtx[idir, :]
        
        if normalize:
            ht = ht/np.amax(ht)

        ax.plot(self.time_ht, ht, label = "Rec #{}".format(idir))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [-]")
        ax.grid()
        ax.set_xlim(xlims)
        
    def ir_raw_vs_windowed(self, idir = 0, normalize = True, xlims = (0, 50e-3)):
        """ Compare same IR before and after windowing
        """
        if normalize:
            ht = self.ht_mtx[idir,:]/np.amax(self.ht_mtx[idir,:])
            htw = self.htw_mtx[idir,:]/np.amax(self.htw_mtx[idir,:])
            
        plt.figure()
        plt.plot(self.time_ht, ht , '-k', label = 'raw', linewidth = 2)
        plt.plot(self.time_ht, htw, '-r', label = 'windowed', alpha = 0.7)
        plt.grid()
        plt.legend()
        plt.xlim(xlims)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [-]")
                
    def plot_all_ir(self, figsize = (15,20), figformat = (7,9),
                    normalize = True, xlims = (0, 50e-3), windowed = False):
        """plot almost all irs
        """
        # Number of curves per axis
        num_of_axis = figformat[0]*figformat[1]
        num_of_cur_axis = int(self.receivers.coord.shape[0]/num_of_axis)
        
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True)
        counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                for curv in range(num_of_cur_axis):
                    self.plot_ir(ax[row, col], idir = counter,
                                 normalize = normalize, xlims = xlims,
                                 windowed = windowed)
                    counter += 1
                if windowed == False:
                    ax[row, col].plot(self.time_ht, self.adrienne_win, 'k')
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                ax[figformat[0]-1, col].set_xlabel("Time [s]")
            ax[row, 0].set_ylabel("Amplitude [-]")
        plt.tight_layout() 
       
    def plot_frf_mag(self, ax, idir = 0, xlims = (20, 20000), ylims = (-150, -20),
                windowed = False, color = None, alpha = None, label = None):
        """ plot an axis with FRF
        """
        if windowed:
            Hw = self.Hww_mtx[idir, :]
        else:
            Hw = np.fft.fft(self.ht_mtx[idir,:])[:self.nfft_half]
        
        ax.semilogx(self.freq_Hw, 20*np.log10(np.abs(Hw)), color = color,
                    alpha = alpha, label = label)
        ax.legend()
        ax.grid(visible=True, which = 'both', linestyle='-')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$|H(f)|$ [dB]")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
    def frf_raw_vs_windowed(self, idir = 0, xlims = (20, 20000), ylims = (-80, -20)):
        """ Compare same FRF before and after windowing
        """     
        fig, ax = plt.subplots(1, figsize = (7,5))
        self.plot_frf_mag(ax, idir = idir, xlims = xlims, ylims = ylims, windowed = False,
                          color = 'b', alpha = 0.5, label = "Raw: Rec #{}".format(idir))
        self.plot_frf_mag(ax, idir = idir, xlims = xlims, ylims = ylims, windowed = True,
                          color = 'm', alpha = 1.0, label = "Windowed: Rec #{}".format(idir))
    
    def plot_all_wfrf(self, figsize = (15,20), figformat = (7,9),
                    xlims = (20, 20000), ylims = (-80, -20)):
        """plot almost all irs
        """
        # Number of curves per axis
        num_of_axis = figformat[0]*figformat[1]
        num_of_cur_axis = int(self.receivers.coord.shape[0]/num_of_axis)
        
        fig, ax = plt.subplots(figformat[0], figformat[1], figsize = figsize,
                               sharex = True, sharey = True)
        counter = 0
        for row in range(figformat[0]):
            for col in range(figformat[1]):
                for curv in range(num_of_cur_axis):
                    self.plot_frf_mag(ax[row, col], idir = counter,
                                 xlims = xlims, ylims = ylims, windowed = True)
                    counter += 1
                
                ax[row, col].set_xlabel("")
                ax[row, col].set_ylabel("")
                ax[figformat[0]-1, col].set_xlabel("Frequency [Hz]")
            ax[row, 0].set_ylabel(r"$|H(f)|$ [dB]")
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