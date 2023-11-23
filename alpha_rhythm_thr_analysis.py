# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 23:10:22 2022

@author: Fedosov
"""



from matplotlib import pyplot as plt


from scipy.io import loadmat
import numpy as np
from scipy.signal import welch, hilbert, firwin


'''
b = firwin(63, 1.0, pass_zero = False, fs = 512)
print(b)


'''


import mne

fs = 512
arr = loadmat('C:/Users/Fedosov/Downloads/probe_noise_supression.mat')

data = arr['eeg_data'] # time x channels, метки - во втором после нулей каналах (4)

plt.figure()
f,pxx = welch(data[:-1000,8],axis = 0, nperseg = fs, fs = 512)
plt.plot(f[3:49],0.5*np.log10(pxx[3:49]))
plt.xlabel('frequency, Hz')
plt.ylabel('psd, dB')
plt.legend(['spectrum of raw signal'])
plt.grid('on')


plt.figure()
filtered = mne.filter.filter_data(data[:-1000,8], fs, 2, 40)
filtered = mne.filter.notch_filter(filtered, fs, [50,150], method = 'fir')
plt.plot(filtered)
plt.ylim(-20000,20000)



plt.figure()
f,pxx = welch(data[:-1000,6],axis = 0, nperseg = fs, fs = 512)
plt.plot(f[3:49],(pxx[3:49])-7.0)
plt.xlabel('frequency, Hz')
plt.ylabel('psd, dB')
plt.legend(['spectrum after iir+kalman filtration on 22 Hz'])
plt.grid('on')








idx_rest = np.zeros((0,512*60),dtype = 'int')
for i in range(1):
    
    idx_rest =np.concatenate((idx_rest,np.arange(0,512*60, dtype = 'int')[None,]+i*512*20))
    
idx_move =  np.zeros((0,512*50),dtype = 'int')
for i in range(1):    
    idx_move =np.concatenate((idx_move,np.arange(0,512*50, dtype = 'int')[None,]+i*512*20+512*60))






plt.figure()
f,pxx = welch(data[idx_rest,8], nperseg = fs, fs = 512,axis = 1)
plt.plot(f[5:40],np.log10(np.mean(pxx[:,5:40],axis = 0)))


f,pxx = welch(data[idx_move,8],nperseg = fs, fs = 512, axis = 1)
plt.plot(f[5:40],np.log10(np.mean(pxx[:,5:40],axis = 0)))





plt.figure()
f,pxx = welch(data[512:512*31,3],axis = 0, nperseg = fs, fs = 512)
plt.plot(f[1:120],np.log10(pxx[1:120]))




#f,pxx = welch(data[512*60:-100,8],axis = 0, nperseg = 1024, fs = 512)
#plt.plot(f[10:60],np.log10(pxx[10:60]))

plt.figure()
filtered = mne.filter.filter_data(data[:-1000,8], fs, 8, 40)
filtered = mne.filter.notch_filter(filtered, fs, [50,150], method = 'fir')
plt.plot(filtered)


#print(data[4100:4300,6]/10000)
for i in range(35):
    plt.figure()
    plt.plot((data[:-100,i]))
    
    #f,pxx = welch(data[:-100,i],axis = 0, nperseg = 512, fs = 512)
    
    #plt.figure()
    #plt.plot(f,np.log10(pxx))
    #plt.plot((data[:-100,10]-0xFFFFF)/1000)

#analyt = hilbert(data, axis =0)

#gt_

