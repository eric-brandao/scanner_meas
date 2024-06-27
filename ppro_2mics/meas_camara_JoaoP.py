# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:28:08 2024

@author: joaop

Codigo pratico para medicoes in situ a serem realizadas na camara reverberante
"""

#%% IMPORTANDO AS BIBLIOTECAS NECESSARIAS
import sys
sys.path.append('C:/WorkspacePython/scanner_meas/scanner')
sys.path.append('C:/Users/joaop/OneDrive/Documents/GitHub/insitu_sim_python/insitu')
import numpy as np
import time

from sequential_measurement import ScannerMeasurement #import do objeto de medicao

#from sequencial_measurement_meta_dados_gustavo_e_joao import ScannerMeasurement


# Receiver class
from receivers import Receiver #import do objeto de recepcao (receptores)

#%% INSTANCIANDO O OBJETO DE MEDICAO
# fs_ni = 51200
name = 'Melamina_14_06_2'
main_folder = 'C:/WorkspacePython/scanner_meas/meas_scripts'# use forward slash
# arduino_dict = dict()
meas_obj = ScannerMeasurement(main_folder = main_folder, name = name,
    fs = 44100, fft_degree = 18, 
    start_stop_margin = [0.1, 0.5], mic_sens = 51.4,
    x_pwm_pin = 2, x_digital_pin = 24,
    y_pwm_pin = 3, y_digital_pin = 26,
    z_pwm_pin = 5, z_digital_pin = 28,
    dht_pin = 40, pausing_time_array = [5, 8, 7])

#%% CHEQUE A LISTA DE INTERFACES DISPONIVEIS

meas_obj.pytta_list_devices()

#%% SELECIONE UMA DAS INTERFACES APRESENTADAS 

meas_obj.pytta_set_device(device = [7,10]) #[7,10] #[15,14]
# 13
#%% GERE UM SWEEP = xt 

meas_obj.set_meas_sweep(method = 'logarithmic', freq_min = 50,
                        freq_max = 20000, n_zeros_pad = 0)


#%% SELECIONE OS CANAIS DE ENTRADA E SAIDA DA INTERFACE

meas_obj.pytta_play_rec_setup(in_channel = 1, out_channel = 1) 

#%% TOCAR SWEEP E VERIFICAR NIVEIS DE IMPUT E OUTPUT

yt_obj = meas_obj.pytta_play_rec()  

#%% ACIONE OS MOTORES DO ROBÔ E CONECTE AO ARDUINO

meas_obj.set_motors()

#%% MOVIMENTACAO UNIAXIAL DO ROBO (ajustes de posicionamento)

meas_obj.move_motor(motor_to_move = 'z', dist = -0.01)

#distance_vector = np.array([0.01, 0.01, 0.01])  # Distâncias para x, y e z
#meas_obj.move_motor_xyz(dist = distance_vector)

#%%# CONFIGURANDO AS POSICOES DO MIC E ARRANJO

# NAO DIGA PARA O ROBO QUE ELE JÁ ESTÁ NO LOCAL CERTO. EXEMPLO: 
# receiver_obj = Receiver(coord = [0, 0, 0.02])
# pt0 = np.array([0.0, 0.0, 0.02]); "--> Coordinates where the michophone is"


receiver_obj = Receiver(coord = [0, 0, 0.02])
# receiver_obj.double_planar_array(x_len=0.1,n_x=2,y_len=0.1,n_y=2, zr=0.015, 
#                                   dz=0.03)
receiver_obj.double_rec(z_dist = 0.02)
#receiver_obj.random_3d_array(x_len = 0, y_len = 0, z_len = -0.02, zr = 0.01, n_total = 1, seed = 0)
# np.random.shuffle(receiver_obj.coord)


pt0 = np.array([0.0, 0.0, 0.01]); "--> Coordinates where the michophone is"
#thickness = 0.05
#pt0[2] = - pt0[2]+thickness # added by Eric to correct the axis movement

meas_obj.set_receiver_array(receiver_obj, pt0 = pt0)

#%% REALIZANDO A MEDICAO

meas_obj.sequential_measurement(meas_with_ni = False,
                                repetitions = 1,
                                plot_frf = True)

#############################################################################