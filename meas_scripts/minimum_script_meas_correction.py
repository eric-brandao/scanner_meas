"""
Created on Thu May 23 2025 - minimum measurement script
"""
# import sys
# sys.path.append('D:/Work/dev/scanner_meas/scanner')
import numpy as np
from IPython.display import clear_output
from sequential_measurement import ScannerMeasurement
from ppro_meas_insitu import InsituMeasurementPostPro
from receivers import Receiver
from sources import Source
import pytta
#%% Measurement name and location
measurement_name = 'AbsTriangOrange_L60cm_d5cm_s100cm_randomarray_576pts_25012025' # Remember good practices --> samplename_arraykeyword_ddmmaaaa
main_folder = 'D:/Work/UFSM/Pesquisa/insitu_arrays/experimental_dataset/working_measurements/RandomArray/'# use forward slash
#%% Load all recorded signals
ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = measurement_name)
rec_signals = ppro_obj.load_allmeas_files()
clear_output()
#### Print info.
ppro_obj.meas_obj.print_meas_data()
#%% Compute PCC of all recorded signals
ppro_obj.pcc_magspk(yt_list = rec_signals, ref_ch = 1)
#%% Flag the bad ones
ppro_obj.flag_measurements(min_pcc = 0.9999, spk_pcc = True)
print("Flagged measurements (indexes): {}".format(ppro_obj.problematic_measurements))
print("Percentual of failure: {:.2f} % ({} of {} measurements)".format(
    100*len(ppro_obj.problematic_measurements)/ppro_obj.meas_obj.receivers.coord.shape[0],
    len(ppro_obj.problematic_measurements), ppro_obj.meas_obj.receivers.coord.shape[0]))
#%% Now we can set up the correction folder
ppro_obj.meas_obj.create_correction_folder()

#%% Check audio devices again (for sound card)
ppro_obj.meas_obj.pytta_list_devices()
#%% Set audio devices again (for sound card)
ppro_obj.meas_obj.pytta_set_device(device = 16)
ppro_obj.meas_obj.pytta_play_rec_setup(in_channel = [1, 3], out_channel = [1, 2],
                         in_channel_ref = 3, in_channel_sensor = 1,
                         output_amplification = -3, repetitions = 1)

#%% Retake the measurements
pt0 = [0, 0, 0.05] # Starting point
ppro_obj.meas_obj.sequential_correction_measurement(
    flagged_measurements_ids = ppro_obj.problematic_measurements,
    pt0 = pt0, bypass_scanner = True, pcc_min = 0.9999, max_num_of_trials = 20)
