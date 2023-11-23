# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:21:43 2023

@author: Fedosov
"""







import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sn

import pickle
import os





class float_int_kalman:
    def __init__(self, freq0, A, srate, q, r):
        
        self.freq0 = freq0
        self.A = A
        self.H = np.array([[1.0,0.0]])
        self.Phi = A*np.array([[np.cos(2.0*np.pi*freq0/srate),-np.sin(2.0*np.pi*freq0/srate)],[np.sin(2.0*np.pi*freq0/srate),np.cos(2.0*np.pi*freq0/srate)]])
        
        
        self.x = np.zeros((2,1))
        self.P = np.eye(2)
        
        self.Q = np.eye(2)*q
        self.R = r
        
        
        
    # y - 2x1
    def step(self, y):
        self.x_ = self.Phi@self.x
        
        self.P_ = self.Phi@self.P@self.Phi.T+self.Q
        
        self.res = y-self.H@self.x_
        
        self.S = self.H@self.P_@self.H.T+self.R
        
        self.K = self.P_@self.H.T/self.S
        
        self.x = self.x_+self.K@self.res
        
        self.P = (np.eye(2)-self.K@self.H)@self.P_
        #print(self.P[0,0])
        
        
    def apply(self, y):
        
        Len = y.shape[0]
        filtered =np.zeros(Len)
        filtered_imag = np.zeros(Len)
        envelope =np.zeros(Len)
        
        for i in range(Len):
            self.step(y[i])
            filtered[i] = self.x[0]
            filtered_imag[i] = self.x[1]
            envelope[i] = np.sqrt(self.x[0]**2+self.x[1]**2)
            
        return filtered, filtered_imag,envelope
    





plt.close('all')


srate = 1000



sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/data1/data.txt")#svetodata.txt")#not_bad_data2.txt")#data.txt")#not_bad_data2.txt")




plt.figure()
plt.plot(sig[:,0])

plt.figure()
plt.plot(sig[:,5])







              
folder_path = 'C:/Users/Fedosov/Documents/projects/mks_return_ICA/results/'
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
      
      # Sort the subfolders by their modified timestamp in descending order
subfolders.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
      
      # Get the name of the subfolder with the latest timestamp
latest_subfolder = os.path.basename(subfolders[0])
      
pathdir = latest_subfolder
      #alpha,beta
      
      

# kalman freq0, A, r, q,ica_filter
file = open(folder_path+pathdir+'/filter_params.pickle', "rb")
container =  pickle.load(file)
file.close()

freq0 = container['freq0']
A = container['A']
q = container['q']
r = container['r']
ica_filter = container['ica_filter']
low_thr = container['low_thr']
high_thr = container['high_thr']

#envelope = np.ones(1)*low_thr


kf = float_int_kalman(freq0, A, srate, 1e-6, 1)





b_dc, a_dc = sn.butter(1,2.0,btype = 'high',fs = srate)

b_high, a_high = sn.butter(1,70.0,btype = 'low', fs = srate)


b50,a50=sn.butter(1,[48.0,52.0], btype = 'bandstop',fs = srate)

b100,a100=sn.butter(1,[97,103], btype = 'bandstop',fs = srate)

b150,a150 = sn.butter(1,[146.0,154.0], btype = 'bandstop',fs = srate)

b200,a200 = sn.butter(1,[195.0,205.0], btype = 'bandstop',fs = srate)

b250,a250 = sn.butter(1,[244.0,256.0], btype = 'bandstop',fs = srate)


filtered =sig[:,0]#sig[:,:5]@ ica_filter
filtered = sn.filtfilt(b_dc,a_dc,filtered)
filtered = sn.filtfilt(b_high,a_high,filtered)
filtered = sn.filtfilt(b50,a50,filtered)
filtered = sn.filtfilt(b100,a100,filtered)
filtered = sn.filtfilt(b150,a150,filtered)
filtered = sn.filtfilt(b200,a200,filtered)
filtered = sn.filtfilt(b250,a250,filtered)



filtered,filtered_imag,_= kf.apply(filtered)

plt.figure()
plt.plot(-filtered_imag)
plt.plot(sig[:,5]/1000)




