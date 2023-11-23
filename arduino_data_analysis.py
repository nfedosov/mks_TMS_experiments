# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:06:06 2023

@author: Fedosov
"""

import numpy as np

import matplotlib.pyplot as plt

import scipy.signal as sn

plt.close('all')
common_data = np.loadtxt('arduino_data.txt')


if common_data[3] == 1111:
    source_signal = common_data[4::5]
    imag_signal = common_data[5::5]
    pink_noise = common_data[6::5]
    read_signal = common_data[7::5]
elif common_data[4] == 1111:
    source_signal = common_data[5::5]
    imag_signal = common_data[6::5]
    pink_noise = common_data[7::5]
    read_signal = common_data[8::5]
elif common_data[5] == 1111:
    source_signal = common_data[6::5]
    imag_signal = common_data[7::5]
    pink_noise = common_data[8::5]
    read_signal = common_data[9::5]
elif common_data[6] == 1111:
    source_signal = common_data[7::5]
    imag_signal = common_data[8::5]
    pink_noise = common_data[9::5]
    read_signal = common_data[10::5]
elif common_data[7] == 1111:
    source_signal = common_data[8::5]
    imag_signal = common_data[9::5]
    pink_noise = common_data[10::5]
    read_signal = common_data[11::5]
    
else:
    pass
    
    
plt.figure()
plt.plot(source_signal)

plt.figure()
plt.plot(imag_signal)

plt.figure()
plt.plot(pink_noise)

plt.figure()
plt.plot(read_signal)




fs =200
f,pxx = sn.welch(source_signal,fs = fs,nperseg = int(2*fs))
plt.figure()
plt.plot(f,np.log10(pxx))






##########
'''
import scipy.stats as st


real_mod = st.zscore(source_signal)
read_mod = st.zscore(read_signal)
plt.figure()
plt.plot(real_mod)
plt.plot(read_mod)


'''

###############



import scipy.signal as sn
fs = 1000
f0 = 10

b,a = sn.butter(1,1.0,'high',fs = fs)

nT = 10000
t = np.arange(nT)/fs
gt = np.sin(2*np.pi*t*f0)
noise = 100

signal= gt+noise

filtered = sn.lfilter(b,a,signal)

plt.figure()
plt.plot(filtered)
plt.plot(gt)









