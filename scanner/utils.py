# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:07:46 2022
utils module for the scanner
@author: ericb
"""
# Imports
import numpy as np
# from scipy import spatial
# from tqdm import tqdm
# from sequential_measurement import ScannerMeasurement
import pytta
# from receivers import Receiver
# from sources import Source
import pickle


def save(obj, filename = 'fname', path = ''):
    """ To save the decomposition object as pickle

    Parameters
    ----------
    filename : str
        name of the file
    pathname : str
        path of folder to save the file
    """
    filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
    path_filename = path + filename + '.pkl'
    f = open(path_filename, 'wb')
    pickle.dump(obj.__dict__, f, 2)
    f.close()

def load(obj, filename = 'fname', path = ''):
    """ To load the decomposition object as pickle

    You can instantiate an empty object of the class and load a saved one.
    It will overwrite the empty object.

    Parameters
    ----------
    filename : str
        name of the file
    pathname : str
        path of folder to save the file
    """
    lpath_filename = path + filename + '.pkl'
    f = open(lpath_filename, 'rb')
    tmp_dict = pickle.load(f)
    f.close()
    obj.__dict__.update(tmp_dict)

def order_closest(pt0, coordinates): # ToDo - on testing it seems to be doing nothing
    """ Order the receiver coordinates
    
    Order the receiver coordinates, so that the new array is a sequence
    of points that are the closest from each other.
    
    Parameters
    ----------
    pt0 : numpy 1dArray;
        3D coordinates of microphone's inicial position/location.
    coordinates : numpy ndArray
        Array with all receiver's coordinates - 'receivers.coord' from the Receiver() object.
    Returns
    -------
    ordened_coord : numpy ndArray
        ordered array

    """
    # initialize
    ordered_coord = np.zeros((coordinates.shape), dtype='float64')
    distance = np.zeros((coordinates.shape[0]), dtype='float64')
    index = np.zeros((coordinates.shape[0]), dtype='int')
    
    all_receivers = list(coordinates) #Listing the original order of receiver points
    pt_center = pt0
    
    # "Index of the closest coordinate from (0, 0, 0):"
    distance[0], index[0] = spatial.KDTree(all_receivers).query(pt_center)  # Finding the closest receiver from pt0
    # Appending the closest point to the new ordened matrix
    ordered_coord[0] = all_receivers[index[0]]
    # After find the closest distance and extract it's coordinates,
    all_receivers[index[0]] = np.array((1e10, 1e10, 1e10), dtype='float64')
    # the coordinates of the original order is changed by a large value
    #"Sorting all points - it considers the closest distance to displace:"
    for ind in range(coordinates.shape[0]-1):
        distance[ind+1], index[ind + 1] = spatial.KDTree(all_receivers).query(ordered_coord[ind])
        ordered_coord[ind+1] = all_receivers[index[ind+1]]
        all_receivers[index[ind+1]] = np.array((1e10, 1e10, 1e10), dtype='float64')
    
    # change of signal in the x-axis (God knows why) # ToDo: Check this
    ordered_coord[:,0] = -ordered_coord[:,0]
    return ordered_coord

def matrix_stepper(pt0, coordinates):
    """ Computes the x,y,z distances to move each motor
    
    Parameters
    ----------
    pt0 : numpy 1dArray;
        3D coordinates of microphone's inicial position/location.
    coordinates : numpy ndArray
        Array with all receiver's (ordered)
    
    Returns
    ----------
    distances_xyz : numpy ndArray;
        x, y, and z distances.
    """
    
    distances_xyz = np.zeros(coordinates.shape)  # mo = np.zeros(rec_coord.shape)
    distances_xyz[0,:] = coordinates[0,:] - pt0
    
    aux0 = coordinates[1:,:] # coordinates from index 1 to end
    aux1 = coordinates[:-1,:] # coordinates from index 0 to end-1
    
    distances_xyz[1:,:] = aux0 - aux1
    
    # change of signal in the y-axis (God knows why) # ToDo: Check this
    distances_xyz[:,1] = -distances_xyz[:,1]    
    
    return distances_xyz

def parse_meas(yt, recs, main_folder = 'D:', name = 'samplename',                 
            fs = 51200, fft_degree = 18, start_stop_margin = [0, 4.5], 
            mic_sens = None):
    """ Parse old measurement to new version
    """
    # set control to save Pytta SigObj
    # meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    #     fs = 51200, fft_degree = 18, start_stop_margin = [0, 4.5], 
    #     mic_sens = 51.4, x_pwm_pin = 2, x_digital_pin = 24,
    #     y_pwm_pin = 3, y_digital_pin = 26, z_pwm_pin = 4, z_digital_pin = 28,
    #     dht_pin = 40, pausing_time_array = [5, 8, 7])

    # receiver_obj = Receiver()
    # receiver_obj.double_rec(z_dist = 0.02)
    # meas_obj.set_receiver_array(receiver_obj, pt0 = np.array([0.0, 0.0, 0.02]))
    
    # meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 1,
    #                     freq_max = (meas_obj.fs)/2, n_zeros_pad = 0)
    
    yt_rec_obj = pytta.classes.SignalObj(signalArray = np.random.normal(0,0.5,2**18), 
                   domain='time', freqMin = 1, freqMax = 25600, samplingRate = 51200)
    # repetitions = yt.shape[1]
    # bar = tqdm(total = 2*yt.shape[1], 
    #            desc = 'Parsing measured signals to a series of hdf5')
    # for jrec in range(2):#range(yt.shape[0]):
    #     for jmeas in range(repetitions):
    #         # Pass to pytta
    #         yt_obj = pytta.classes.SignalObj(signalArray = np.random.normal(0,0.5,2**18), 
    #               domain='time')
    #         # # ptta saving
    #         # filename = 'rec' + str(int(jrec)) +\
    #         #     '_m' + str(int(jmeas)) + '.hdf5'
    #         # complete_path = meas_obj.main_folder / meas_obj.name / 'measured_signals'
    #         # pytta.save(str(complete_path / filename), yt_obj)
    #         bar.update(1)
    # bar.close()

def wtf(yt = 3):
    print(type(yt))
    yt_rec_obj = pytta.classes.SignalObj(signalArray = yt, 
                   domain='time', freqMin = 1, freqMax = 25600, samplingRate = 51200)
    return yt_rec_obj