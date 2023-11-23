# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:35:12 2023

@author: Fedosov
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:42:35 2023

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
        
        
    def apply(self, y, moments):
        
        Len = y.shape[0]
        filtered_real =np.zeros(Len)
        filtered_imag = np.zeros(Len)
        envelope =np.zeros(Len)
        phase = np.zeros(Len)
        
        
        
        refract_T = 330
        counter = 0
        for i in range(Len):
            if moments[i] == 1.0:
                counter = 0
            if counter > refract_T:
                self.step(y[i])
            counter += 1
                
            filtered_real[i] = self.x[0]
            filtered_imag[i] = self.x[1]
            envelope[i] = np.sqrt(self.x[0]**2+self.x[1]**2)
            phase[i] = np.angle(self.x[0]+self.x[1]*(1j),deg = True)
            
        return filtered_real, filtered_imag,envelope, phase
    












#from kalman_float_int import float_int_kalman,  simplified_kf

plt.close('all')


srate = 1000


#ica_filter = np.array([-168,   46, -316,  -38,  244,   75,   -6,  255,  -68])


sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/raw_data/data_arbus.txt")#svetodata.txt")#not_bad_data2.txt")#data.txt")#not_bad_data2.txt")

'''
b, a = sn.butter(1,1.955,btype = 'high',fs = srate)

#b, a = (np.round(0x1000*b)).astype(int),(np.round(0x1000*a)).astype(int)

filtered_sig = sn.filtfilt(b,a,sig[:,0])
plt.figure()
plt.plot(filtered_sig)

'''


if (0):
    for i in range(6):
        plt.figure()
        plt.plot(sig[:,i])
        plt.ylabel('magnitude')
        plt.xlabel('time, ms')


plt.close('all')
plt.figure()
plt.plot(sig[1000:,5]*1e6)
plt.plot(sig[1000:,0])
plt.title('Trigger signal and artefact response signal on EEG')
plt.xlabel('time, ms')
plt.ylabel('magnituge')

'''
for i in range(8):
    plt.figure()
    plt.plot(sig[1000:,i])'''
    
    
'''  
prefiltered = sig[100:,-2]

f,pxx = sn.welch(sig[58000:70000,0], fs = 1000, nperseg = 1000)
plt.figure()
plt.plot(f,np.log10(pxx))'''


'''
import mne
mne.create_info(['1','2','3','4','5'], sfreq = 1000, ch_types='EEG')

raw = mne.io.RawArray(sig[:,:5], info)
'''


'''
plt.figure()
plt.plot(sig[1000:,-1])
plt.figure()
plt.plot(sig[1000:,0])
plt.figure()
plt.plot(sig[1000:,1])
plt.figure()
plt.plot(sig[1000:,2])
plt.figure()
plt.plot(sig[1000:,3])
plt.figure()
plt.plot(sig[1000:,4])

'''


'''
#ica = sig[1000:,:4]@ica_filter_int[:,None]
ica = np.array([sig[1000:,0],sig[1000:,1],sig[1000:,2],sig[1000:,3]]).T@ica_filter_int[:,None]
ica = ica[:,0]

b,a = sn.butter(3,[8.0,13.0],fs = 1000, btype = 'bandpass')
b50,a50 = sn.butter(3,[48.0,52.0],fs = 1000,btype = 'bandstop')
ica = sn.lfilter(b,a,ica)
ica = sn.lfilter(b50,a50,ica)


plt.figure()
plt.plot(ica)
'''

'''
plt.figure()
plt.plot(sig[1000:,-1]-np.roll(sig[1000:,0],16))



print((sig[1000:,-1]-np.roll(sig[1000:,0],16))[-100:])

'''


moments = np.zeros(len(sig[:,0]))

#thr = 5.560e6
#refract_T = 1000



'''
for  i in range(len(sig[:,0])):
    
    if (sig[i,-1] >thr) and (t > refract_T):
        moments[i] = 1
        t = 0
        
    t += 1'''
    
where_all_zeros = np.sum(sig[:,:5],axis = 1)
all_zeros = (where_all_zeros == 0)
for i in range(1,len(sig[:,0])):
    
    
    if (all_zeros[i] == True) and ((all_zeros[i-1] == False)):
        moments[i] = 1
    
    
        
    
    
    
    
    
plt.figure()
plt.plot(moments*1e9)
plt.plot(sig[:,-1])
    

times = np.where(moments)[0]
times = times[1:-1]


idx = times[:,np.newaxis]+np.arange(-500,500)[np.newaxis,:]


epochs = sig[idx,-3]


types = sig[times,-1]
target_idx = (types==9)

plt.figure()
plt.plot(np.arange(-500,500),epochs[target_idx,:].T)
plt.xlabel('times, ms')
plt.ylabel('magnitude')

plt.plot(np.array([0,0]),np.array([-5e6,5e6]),'--', color = 'red')




plt.figure()
plt.plot(np.arange(-500,500),np.mean(epochs[target_idx],axis = 0))
plt.plot(np.zeros((2,)),np.array([-100000,100000]))
plt.xlabel('times, ms')
plt.ylabel('magnitude')
        



              
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
filtered[filtered == 0] = np.mean(filtered[np.setdiff1d(np.arange(len(filtered)),np.where(filtered == 0)[0])])
filtered = sn.lfilter(b_dc,a_dc,filtered)
filtered = sn.lfilter(b_high,a_high,filtered)
filtered = sn.lfilter(b50,a50,filtered)
filtered = sn.lfilter(b100,a100,filtered)
filtered = sn.lfilter(b150,a150,filtered)
filtered = sn.lfilter(b200,a200,filtered)
filtered = sn.lfilter(b250,a250,filtered)



filtered_real,filtered_imag,envelope,phase= kf.apply(filtered, moments = moments)





plt.figure()
plt.plot(filtered_imag)
plt.plot(sig[:,5]/1000000)



phase[phase<0] = phase[phase<0]+360
epochs = phase[idx]
target_idx = (types==9)
plt.figure()
plt.plot(np.arange(-500,500),epochs[target_idx,:].T)




plt.figure()
plt.plot(np.arange(-500,500),np.mean(epochs,axis = 0))
plt.plot(np.zeros((2,)),np.array([-5,5]))
        


from scipy.stats import circstd

halfwin = 500

cum_circ_std  = 0


for i,tar_phase in enumerate(range(0,360,20)):
    cum_circ_std += circstd(np.min([360-np.abs(tar_phase-epochs[target_idx,500]),
                                   np.abs(tar_phase-epochs[target_idx,500])],axis = 0)*np.pi/180)
    
    
cum_circ_std /= 18
    
    
print(cum_circ_std*180/np.pi)



plt.figure()
plt.plot(sig[:,0])

plt.xlabel('time, ms')
plt.ylabel('magnitude')











    
