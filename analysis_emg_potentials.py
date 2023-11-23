# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:02:41 2023

@author: Fedosov
"""

import mne
import numpy as np
import pandas as pd

import os



T_win = 0.3
 





directory = 'C:/Users/Fedosov/Downloads/Pilot 3/Pilot 3'
dir_to_save = 'C:/Users/Fedosov/Downloads/Pilot 3/MEPs'
#N_bukv = len(directory)
for filename in os.listdir(directory):
    if filename.endswith('.edf'):
        print(filename)
           
        
        
       
        raw = mne.io.read_raw_edf(os.path.join(directory,filename),preload = True)
        srate = raw.info['sfreq']
        
        N_win= int(T_win*srate)
        
        stims = raw[:,:][0][3,:]<-0.2
        
        
        raw.notch_filter([50.0,100.0])
        raw.filter(0.1,200)
        
        emg = raw[:,:][0][1:3,:]
        
        
        
        meps_APB = list()
        meps_ADM = list()
        times =list()
        
        nt_points = np.where(stims)[0]
        
        
        
        
        for i in nt_points:
            emg_win_APB = emg[0,i:i+N_win]
            emg_win_ADM= emg[1,i:i+N_win]
            
            meps_APB.append(np.abs(np.max(emg_win_APB)-np.min(emg_win_APB)))
            meps_ADM.append(np.abs(np.max(emg_win_ADM)-np.min(emg_win_ADM)))
            times.append(i/srate)
            
            
        #raw = raw.pick([1,])
        
        #fname_to_save = filename[:N_bukv]
        fname_to_save = filename[:-4]
        fname_to_save = os.path.join(dir_to_save,fname_to_save+'.csv')
        
        
        join_np_array = np.array([times,meps_APB,meps_ADM]).T
        
        df = pd.DataFrame(join_np_array)
        
        df.to_csv(fname_to_save, index=False,header=['time [s]','APB [uV]','ADM [uV]'])
        
        
        
    
    
    
    
    




