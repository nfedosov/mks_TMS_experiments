# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 04:00:28 2023

@author: Fedosov
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:59:51 2023

@author: Fedosov
"""

from lsl_inlet import LSLInlet
import numpy as np
import scipy.signal as sn
import pickle
import os





  
        
exp_settings = {
    #максимальная буферизация принятых данных через lsl
    'max_buflen': 5,  # in seconds!!!!
    #максимальное число принятых семплов в чанке, после которых появляется возможность
    #считать данные
    'max_chunklen': 1,  # in number of samples!!!!
    'lsl_stream_name': 'AAA'
    }
inlet = LSLInlet(exp_settings)
inlet.get_frequency()
        
        
        #!!!!
srate= 1000
n_channels = 4
        
 
data= np.zeros((60*1000,n_channels))
        
n_samples_received = 0

   
counter = 0     
while(counter <1000):
    
    
    
    chunk, t_stamp = inlet.get_next_chunk()
        # print(f"{chunk=}")
    if chunk is not None:
        n_samples_in_chunk = len(chunk)
        data[n_samples_received:n_samples_received + n_samples_in_chunk, :] = chunk
        n_samples_received += n_samples_in_chunk
        counter += 1
        
     
from matplotlib import pyplot as plt
 
for i in range(4):
    plt.figure()
    plt.plot(data[:,i])
        


