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
import scipy.stats as st











#from kalman_float_int import float_int_kalman,  simplified_kf

plt.close('all')


srate = 1000


#ica_filter = np.array([-168,   46, -316,  -38,  244,   75,   -6,  255,  -68])


#sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/raw_data/data_arbus_04102023_3exp_separate_source.txt")

sig = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/raw_data/data_arbuz_10102023_1exp_low_noise_hard_filter.txt")#data.txt")#not_bad_data2.txt")


common_data = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/arduino_data/arduino_data_arbuz_10102023_1exp_no_noise_hard.txt")

#common_data = np.loadtxt("C:/Users/Fedosov/Documents/projects/mks_return_ICA/arduino_data/arduino_data_arbuz_04102023_3_exp_separate_source.txt")








'''
if common_data[3] == 1111:
    source_signal = common_data[4::4]
   
    pink_noise = common_data[5::4]
    read_signal = common_data[6::4]
elif common_data[4] == 1111:
    source_signal = common_data[5::4]
   
    pink_noise = common_data[6::4]
    read_signal = common_data[7::4]
elif common_data[5] == 1111:
    source_signal = common_data[6::4]
    
    pink_noise = common_data[7::4]
    read_signal = common_data[8::4]
elif common_data[6] == 1111:
    source_signal = common_data[7::4]
    
    pink_noise = common_data[8::4]
    read_signal = common_data[9::4]
'''


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
    
    

    
    
    



'''
b, a = sn.butter(1,1.955,btype = 'high',fs = srate)

#b, a = (np.round(0x1000*b)).astype(int),(np.round(0x1000*a)).astype(int)

filtered_sig = sn.filtfilt(b,a,sig[:,0])
plt.figure()
plt.plot(filtered_sig)

'''


if (0):
    plt.close('all')
    for i in range(8):
        plt.figure()
        plt.plot(sig[:,i])
        plt.ylabel('magnitude')
        plt.xlabel('time, ms')
        
        
############# IF INVERTED!!!!
source_signal *= -1
imag_signal *= -1



t_range_for_psd = [560000,610000]# [560000, 568000] for 0410exp

raw_for_psd = sig[t_range_for_psd[0]:t_range_for_psd[1],0]
plt.figure()
plt.plot(raw_for_psd)
plt.xlabel('time, ms')

f,pxx = sn.welch(raw_for_psd,fs = srate, nperseg = int(srate/2))


plt.figure()
plt.plot(f,np.log10(pxx))
plt.xlabel('Hz')
plt.ylabel('log(pxx)')


#f,pxx = sn.welch(source_signal/2.5+pink_noise*10,fs = srate, nperseg = int(srate*2))


#plt.figure()
#plt.plot(f,np.log10(pxx))





plt.figure()
plt.plot(read_signal)


arduino_trigger = (read_signal<400.0)*1.0
arduino_moments = ((arduino_trigger[1:]-arduino_trigger[:-1]) > 0)*1.0

#warning!!!!
arduino_timepoints = np.where(arduino_moments)[0][1:]


plt.figure()
plt.plot(arduino_trigger)
plt.plot(arduino_moments)






plt.figure()
plt.plot(source_signal)
plt.plot(arduino_moments)

source_signal_modified = source_signal.copy()


###
b,a = sn.butter(1,[30],btype = 'low',fs = 200)
source_signal_modified = sn.filtfilt(b,a,source_signal_modified)
###


cmplx_gt = sn.hilbert(source_signal_modified)
imag_gt = np.imag(cmplx_gt)
plt.figure()
plt.plot(source_signal_modified)
plt.plot(imag_gt)


gt_phase = np.angle(cmplx_gt)
plt.figure()
plt.plot(gt_phase)




target_sig = sig[::5,-1]


recon_arduino_moments = np.zeros(len(target_sig))

arduino_timepoints_corrected = arduino_timepoints-2000
recon_arduino_moments[arduino_timepoints_corrected]= 1

plt.figure()
plt.plot(target_sig)
plt.plot(recon_arduino_moments*5)


target_phases = target_sig[arduino_timepoints_corrected]*20

real_phases = gt_phase[arduino_timepoints]*180/np.pi
real_phases[real_phases<0] += 360

plt.figure()
plt.plot(target_phases)
plt.plot(real_phases)



def phase_difference(angle1, angle2):
    # Ensure angles are in the range [0, 2π]
    #angle1 = angle1 % (2 * math.pi)
    #angle2 = angle2 % (2 * math.pi)

    # Calculate the absolute angular difference
    abs_diff = abs(angle1 - angle2)

    # Consider the "wrap-around" case when angles are close to 0 and 2π
    abs_diff[abs_diff>180] = 360-abs_diff[abs_diff>180]
    

    return abs_diff


phase_dif = phase_difference(target_phases, real_phases)#st.circstd(target_phases, real_phases)



std_dif = st.circstd(phase_dif*np.pi/180)

plt.figure()
plt.hist(phase_dif,bins = 20, range = [0, 180])


sign_phase_dif = real_phases-target_phases

plt.figure()
plt.hist(sign_phase_dif,bins = 20, range = [-360, 360])












########## PROOF OF HILBERT
'''
plt.close('all')
fs =200
arg = 2*np.pi*10/fs
Phi = 0.99*np.array([[np.cos(arg),-np.sin(arg)],[np.sin(arg),np.cos(arg)]])
x = np.zeros((2,1))

y = np.zeros((10000,2))
for i in range(10000):
    x = Phi@x+np.random.randn(2,1)
    y[i,:] = x[:,0]
plt.figure()
plt.plot(y)

recon = np.imag(sn.hilbert(y[:,0]))
plt.plot(recon)

'''
##########



#moments = np.zeros(len(sig[:,0]))

#thr = 5.560e6
#refract_T = 1000



'''
for  i in range(len(sig[:,0])):
    
    if (sig[i,-1] >thr) and (t > refract_T):
        moments[i] = 1
        t = 0
        
    t += 1'''
    
    
    
    
    
    
    
    
    
    
    
plt.figure() 
plt.plot(sig[::5,5])   

plt.plot(arduino_moments[1000:]*1000000)


 