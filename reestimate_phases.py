# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:51:20 2023

@author: Fedosov
"""



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


sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/raw_data/data1half_0808.txt")#svetodata.txt")#not_bad_data2.txt")#data.txt")#not_bad_data2.txt")

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
plt.plot(moments*1e1)
plt.plot(sig[:,-1])
    

times = np.where(moments)[0]



















































              
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
#file = open('results/baseline_experiment_07-20_11-15-52/filter_params.pickle', "rb")
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




### FOR HIGH POLARIZATION IT IS CHANED!!
b_dc, a_dc = sn.butter(3,2.0,btype = 'high',fs = srate)

b_high, a_high = sn.butter(1,70.0,btype = 'low', fs = srate)


b50,a50=sn.butter(1,[48.0,52.0], btype = 'bandstop',fs = srate)

b100,a100=sn.butter(1,[97,103], btype = 'bandstop',fs = srate)

b150,a150 = sn.butter(1,[146.0,154.0], btype = 'bandstop',fs = srate)

b200,a200 = sn.butter(1,[195.0,205.0], btype = 'bandstop',fs = srate)

b250,a250 = sn.butter(1,[244.0,256.0], btype = 'bandstop',fs = srate)


new_sig = np.zeros_like(sig)[:,0]
bad_idx = np.where(all_zeros)[0]
good_idx = np.setdiff1d(np.arange(len(all_zeros)),bad_idx)
filtered =sig[good_idx,:5]@ica_filter
filtered[filtered == 0] = np.mean(filtered[np.setdiff1d(np.arange(len(filtered)),np.where(filtered == 0)[0])])
filtered = sn.lfilter(b_dc,a_dc,filtered)
filtered = sn.lfilter(b_high,a_high,filtered)
filtered = sn.lfilter(b50,a50,filtered)
filtered = sn.lfilter(b100,a100,filtered)
filtered = sn.lfilter(b150,a150,filtered)
filtered = sn.lfilter(b200,a200,filtered)
filtered = sn.lfilter(b250,a250,filtered)


new_sig[good_idx] = filtered

filtered = new_sig


filtered_real,filtered_imag,envelope_float, phase= kf.apply(filtered,moments)

plt.figure()
plt.plot(filtered_imag)
plt.plot(sig[:,5]/10000000)


plt.figure()
plt.plot(filtered_real)
plt.plot(filtered_imag)
plt.plot(envelope_float)
plt.plot(moments*100)

plt.figure()
plt.plot(phase)




phase_std = 5  #+-5 grad
phase_list = np.arange(-180,180,20)

phase_store = np.zeros(len(phase_list))


moments_idx = np.where(moments)[0]
for i in range(len(moments_idx)):
    where_it_goes = np.abs(phase[moments_idx[i]]-phase_list) < 5
    
    where_it_goes += 360-np.abs(phase[moments_idx[i]]-phase_list) < 5
    
    where_it_goes = where_it_goes > 0
    
    if np.sum(where_it_goes>1):
        print('ERROR PHASE DIFF!!!')
        
    phase_store[where_it_goes] += 1
    
plt.figure()
plt.bar(phase_list, phase_store, width = 10)


#%%
plt.close('all')

import mne


# fouth rec - 'C:/Users/Fedosov/Downloads/Pilot 2/Belocopytov Anton_2021_08_07_18_17_20EMG.edf'
# third rec - 'C:/Users/Fedosov/Downloads/Pilot 2/Belocopytov Anton_2021_08_07_16_49_06EMG.edf'
#first rec -    raw_data/EEG-bci/Anton Belokotytov_2021_07_19_13_27_53EMG.edf
#second rec- raw_data/EEG-bci/Anton Belokotytov_2021_07_19_13_46_04EMG.edf



raw_emg = mne.io.read_raw_edf('C:/Users/Fedosov/Downloads/Pilot 2/Belocopytov Anton_2021_08_07_16_49_06EMG.edf',preload = True)


trigger_3000 = raw_emg[2,:][0][0]

#convolve first
trigger_3000 = np.convolve(trigger_3000,np.ones(3),mode ='same')
trigger = trigger_3000[::3]

plt.figure()
plt.plot(trigger)


raw_emg.resample(1000.0)

#raw_emg.notch_filter([50.0,100.0,150.0,200.0])
#raw_emg.filter(l_freq = None, h_freq = 100.0)
raw_emg.plot()


emg = raw_emg[1,:][0][0]







plt.figure()
plt.plot(emg.T)

plt.figure()
plt.plot(trigger.T)
plt.plot(moments)

# align with eeg recordings


v = np.sin(np.arange(21)/20*2*np.pi)

moments_conv = np.convolve(moments, v)
trigger_conv = np.convolve(-trigger, v)

# for second rec: moments_conv[300:]

cor_res = np.correlate(trigger_conv,moments_conv[:int(len(moments_conv)*9/10)],mode = 'valid')
plt.figure()
plt.plot(cor_res)

bias = np.argmax(cor_res)


# for second i-bias +300

# первая версия, без конкретных фаз

phase_arr = list()
emg_amp = list()
err_arr = list()
#emg indices, encoutering the EEG recording
for i in range(bias-100,len(moments)+100):
    if trigger[i]<-0.20:
        #оцениваем,  соответствующую фазу
        
        err_window = np.arange(-200,0)+i-bias
        
        klmn_seg = filtered_real[err_window]
        klmn_seg -= np.mean(klmn_seg)
        orig_seg = filtered[err_window]
        orig_seg -= np.mean(orig_seg)
        
        err_arr.append(-np.sum(klmn_seg*orig_seg)/(np.linalg.norm(klmn_seg)*np.linalg.norm(orig_seg)))
        
        #(filtered - filtered_real)/np.

        
        phase_arr.append(phase[i-bias])
        
        emg_win = np.arange(0,100)+i
        
        min_amp = np.min(emg[emg_win])
        max_amp = np.max(emg[emg_win])
        
        emg_amp.append(np.abs(min_amp-max_amp))
        
        

plt.figure()
plt.scatter(phase_arr,np.log10(emg_amp))



median = np.quantile(err_arr,0.8)
plt.figure()
plt.scatter(np.array(phase_arr)[np.array(err_arr)<median],
            np.log10(np.array(emg_amp)[np.array(err_arr)<median]))



np.save('phase_and_emg_and_err_3.npy', (phase_arr,emg_amp,err_arr))

container = np.load('phase_and_emg_and_err_3.npy')
ph =container[0]
am = container[1]
err = container[2]










































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





'''







moments = np.zeros(len(sig[:,0]))

#thr = 5.560e6
#refract_T = 1000




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




plt.figure()
plt.plot(np.arange(-500,500),np.mean(epochs[target_idx],axis = 0))
plt.plot(np.zeros((2,)),np.array([-100000,100000]))
plt.xlabel('times, ms')
plt.ylabel('magnitude')
        

'''




    





