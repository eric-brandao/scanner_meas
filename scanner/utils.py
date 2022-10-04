# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:07:46 2022
utils module for the scanner
@author: ericb
"""
# Imports
import numpy as np
from scipy import spatial



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