# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:02:41 2023

@author: Fedosov
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


plt.close('all')
T_win = 0.3
 

path = 'C:/Users/Fedosov/Downloads/dmitriy mitchenkov_2021_11_22_13_51_33EMG.edf'



        

   
raw = mne.io.read_raw_edf(path,preload = True)
srate = raw.info['sfreq']

N_win= int(T_win*srate)

stims = raw[:,:][0][4,:]<-0.2

plt.figure()
plt.plot(raw[:,:][0][4,:])

raw.notch_filter([50.0,100.0])
raw.filter(0.1,200)

emg = raw[:,:][0][2,:]




times =list()

nt_points = np.where(stims)[0]

meps = list()


for i in nt_points:
    emg_win = emg[i:i+N_win]
 
    meps.append(np.abs(np.max(emg_win)-np.min(emg_win)))
  
    times.append(i/srate)
    
    
np_meps = np.array(meps[1:])
    

        
    
sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/raw_data/data_0and180phases100trials.txt")#svetodata.txt")#not_bad_data2.txt")#data.txt")#not_bad_data2.txt")


phase_label = np.array([9, 0, 0, 0, 9, 9, 0, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9,
       0, 0, 0, 9, 9, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 9, 0, 0, 0, 9, 9, 9,
       0, 9, 0, 0, 0, 9, 0, 9, 0, 9, 9, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 9,
       9, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 9, 9, 0,
       0, 9, 0, 9, 0, 9, 9, 9, 0, 9, 9, 9],dtype = 'int')





idx_0 = phase_label==0

idx_180 = phase_label==9



log_meps =np.log10(np_meps)




log_meps_0 = log_meps[idx_0]
log_meps_180 = log_meps[idx_180]




plt.figure()


plt.hist(log_meps_0, bins=10, alpha = 0.5, range = [-4.7,-2.0])
plt.hist(log_meps_180, bins=10, alpha = 0.5, range = [-4.7,-2.0])

print(np.mean(log_meps_0),np.mean(log_meps_180))





    
    






