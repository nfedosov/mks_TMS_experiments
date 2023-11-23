

import mne
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import scipy.signal as sn
import pickle





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
    



'''


class ExponentialSmoother:
    def __init__(self, factor):
        self.a = [1, -factor]
        self.b = [1 - factor]
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1,))

    def apply(self, chunk: np.ndarray):
        y, self.zi = sn.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y

'''



#from scipy.signal import bytter, hilbert,filtfilt
def ideal_envelope(fc,srate, y, bandwidth = 4.0):
    
    b,a = sn.butter(1,[fc-bandwidth/3.0,fc+bandwidth/3.0], fs = srate, btype = 'bandpass')
    
    filtered = sn.filtfilt(b,a,y)
    
    hilb = sn.hilbert(filtered)
    
    envelope = np.abs(hilb)
    
    return envelope



plt.close('all')

np.random.seed(0)


folder_path = 'C:/Users/Fedosov/Documents/projects/mks_return_ICA/results/'
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

# Sort the subfolders by their modified timestamp in descending order
subfolders.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)

# Get the name of the subfolder with the latest timestamp
latest_subfolder = os.path.basename(subfolders[0])




pathdir = latest_subfolder
#alpha,beta




file = open(folder_path+pathdir+'/data.pickle', "rb")
container =  pickle.load(file)
file.close()

exp_settings = container['exp_settings']
srate = exp_settings['srate']
data = container['eeg']
stims = container['stim']
channel_names = exp_settings['channel_names']

# 5 CHANNELS !!!!!!!!!!!!!!!!!!!!!!!!!!! NUMBER OF CHANNELS

for i in range(5):
    channel_names[i] = channel_names[i].upper()
print(channel_names)
n_channels = len(channel_names)




#transfrom to mne format

# for test

####


####


info = mne.create_info(ch_names=channel_names, sfreq = srate, ch_types = 'eeg',)
raw =  mne.io.RawArray(data.T, info)




print(data.shape)
print(info)

### ASSUME THAT MOVE AND REST DURATIONS ARE EQUAL
duration = exp_settings['blocks']['Open']['duration']
open_id = exp_settings['blocks']['Open']['id']
close_id = exp_settings['blocks']['Close']['id']
prepare_id = exp_settings['blocks']['Prepare']['id']
ready_id = exp_settings['blocks']['Ready']['id']



start_times = (np.where(np.isin(stims[1:]-stims[:-1], [open_id -prepare_id,close_id-prepare_id]))[0]+1)/srate



description = []
for st in start_times:
    if stims[int(round((st*srate)))] == open_id:
        description.append('Open')

    if stims[int(round((st*srate)))] == close_id:
        description.append('Close')

    #if stims[int(round((st*srate)))] == prepare_id:
    #    description.append('Prepare')

#description =  stims[(start_times*srate).astype(int)].astype(str)
print(description, start_times, duration)




annotations = mne.Annotations(start_times, duration, description)
raw.set_annotations(annotations)


#montage = mne.channels.make_standard_montage('standard_1005')
#raw.set_montage(montage)




raw_copy = raw.copy()

#HERE MANUALLY MARK BAD SEGMENTS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




#raw.plot_psd()

#input('Press <Space> if you have marked bad segments...')

bad_segments = []

bad_segments.append(np.arange(5*srate))
for annot in raw._annotations:
    if annot['description'] == 'BAD_':
        #print('HERE')
        bad_segments.append(np.arange(int(round(annot['onset']*srate)),int(round((annot['onset']+annot['duration'])*srate))).tolist())


last_idx = int(round(raw[:][1][-1]*srate))

bad_segments.append(np.arange(last_idx-5*srate,last_idx+1))
bad_segments =np.concatenate(bad_segments)
good_idx = np.setdiff1d(np.arange(last_idx+1),bad_segments)






raw.notch_filter([50.0,100.0,150.0,200.0])


raw_2 = raw.copy()
raw.filter(2.0,30.0)

raw_for_vis = raw.copy()
raw_for_vis.filter(0.5,70.0)
raw_for_vis.plot(scalings = dict(eeg=1e-4))

plt.show()
raw_2.filter(0.5,30.0)


ica = mne.preprocessing.ICA()
ica.fit(raw, start = int(5*srate), stop = int(last_idx-5*srate))

ics = ica.get_sources(raw_2)
ica.plot_sources(raw_2)

events = mne.events_from_annotations(raw)

ics_open = mne.Epochs(ics,events[0], tmin = 0, tmax = duration, event_id = events[1]['Open'], baseline = None)
ics_close = mne.Epochs(ics,events[0], tmin = 0, tmax = duration, event_id = events[1]['Close'],baseline = None)

rel_alphas =np.zeros(n_channels)
central_freq_alphas= np.zeros(n_channels)

fig, axs = plt.subplots(3, 3)
for i in range(n_channels):

    ics_open.plot_psd(ax = axs[i//3, i%3], picks = [i,], color = 'red', spatial_colors = False,fmin = 2.0,fmax = 30)
    ics_close.plot_psd(ax = axs[i//3, i%3], picks = [i,], color = 'blue', spatial_colors = False,fmin  = 2.0,fmax = 30)
    axs[i//3, i%3].set_title(str(i))
    
    psd_alpha_open = ics_open.compute_psd(fmin = 9.0, fmax = 13.0,picks = [i,]).get_data(return_freqs = True)
   
    psd_alpha_close = ics_close.compute_psd(fmin = 9.0, fmax = 13.0,picks = [i,]).get_data(return_freqs = True)
 
    rel_alphas[i] = np.mean(psd_alpha_close[0])/np.mean(psd_alpha_open[0])
  
    central_freq_alphas[i]= psd_alpha_close[1][np.argmax(np.mean(psd_alpha_close[0],axis = 0))]


plt.show()





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

str_idx = input('write an integer idx of SMR component...\n')
alpha_idx = int(str_idx)
#beta_idx = 3
print('index of SMR component is ', alpha_idx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#central_alpha = float(input('write alpha central frequency...\n'))
#central_beta = float(input('write beta central frequency...\n'))

central_alpha = central_freq_alphas[alpha_idx]
print(central_alpha)


#????? можно сделать тоже вводимыми значениями
bands = [[central_alpha - 2.0, central_alpha + 2.0],[20-3.0,20+3.0]]




pca_M = ica.pca_components_

ica_M = ica.unmixing_matrix_

unmixing_M = ica_M@pca_M
#unmixing_M = ica.filters_.T


ica_filter = unmixing_M[alpha_idx,:]
#beta_ica_filter = alpha_ica_filter.copy()

#alpha_rhythm = ics[beta_idx,:][0][0,:]
#beta_rhythm = alpha_rhythm.copy()


#plt.figure()
#plt.plot(alpha_rhythm)
#plt.plot(ica_filter@raw[:][0]*0.2e6)


#bad_ica = [] # WRITE HERE BAD ICA numbers
#I = np.eye(n_channels)
#I[bad_ica,bad_ica] = 0
#filtering_matrix = np.linalg.inv(unmixing_M)@ I@ unmixing_M




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


raw_realt = raw_copy[:][0]

# q_s, r_s - ????
freq_alpha = (bands[0][1]+bands[0][0])/2



b_low_dcm, a_low_dcm = sn.butter(2,50.0,btype = 'low',fs = 10000)
print('DCM COEFS')
print(np.round(b_low_dcm*0x2000).astype('int'))
print(np.round(a_low_dcm*0x2000).astype('int'))
print()
'''
b_low_dcm_int, a_low_dcm_int = np.round(b_low_dcm*0x2000).astype('int'),np.round(a_low_dcm*0x2000).astype('int')

noise = np.random.randn(100000)
filtered1 = sn.lfilter(b_low_dcm,a_low_dcm,noise)
filtered2 = sn.lfilter(b_low_dcm_int/0x2000,a_low_dcm_int/0x2000,noise)

plt.figure()
plt.plot(filtered1)
plt.plot(filtered2)
'''





b_dc, a_dc = sn.butter(1,2.0,btype = 'high',fs = srate)

#b_low, a_low = sn.butter(1,50.0,btype = 'low',fs = srate)

b50,a50=sn.butter(1,[48.0,52.0], btype = 'bandstop',fs = srate)

b100,a100=sn.butter(1,[97,103], btype = 'bandstop',fs = srate)

b150,a150 = sn.butter(1,[146.0,154.0], btype = 'bandstop',fs = srate)

b200,a200 = sn.butter(1,[195.0,205.0], btype = 'bandstop',fs = srate)

b250,a250 = sn.butter(1,[244.0,256.0], btype = 'bandstop',fs = srate)



#???

ica_filter_norm = ica_filter*np.sqrt(n_channels)/np.linalg.norm(ica_filter)
realt = ica_filter_norm@raw_realt



realt = sn.lfilter(b50,a50,realt)
realt = sn.lfilter(b50,a50,realt)
realt = sn.lfilter(b100,a100,realt)
realt = sn.lfilter(b150,a150,realt)
realt = sn.lfilter(b200,a200,realt)
realt = sn.lfilter(b250,a250,realt)
realt = sn.lfilter(b_dc,a_dc,realt)




#gt_alpha_envelope = ideal_envelope(freq_beta, srate, realt)#[good_idx]


#np.abs(cfir_filtered)

###### GT THE SAME

#gt_alpha_envelope = envelope_alpha.copy()
#kalman approach
A = 0.992
q = 0.01
r = 100
kf = float_int_kalman(freq_alpha,A,srate,q,r)
filtered, _, envelope_alpha = kf.apply(realt)


'''
plt.figure()
plt.plot(np.arange(1,100)*1000/srate,cc_cum)
plt.xlabel('latency, ms')
plt.ylabel('corr')
'''
#plt.plot([20,20],[0.55,0.9])

'''
tosmooth = 1
if tosmooth:
    b_smooth, a_smooth = sn.butter(1,0.31,btype = 'low',fs = srate)
    print('smoothing filter')
    print(np.round(b_smooth*0x1000).astype('int'))
    print(np.round(a_smooth*0x1000).astype('int'))
    envelope_alpha = sn.lfilter(b_smooth,a_smooth,envelope_alpha)
'''


thr_low,thr_high = np.quantile(envelope_alpha[good_idx],[0.1,0.7])
thr_low = 0#-thr_high/2

'''
if tosmooth:
    thr_low,thr_high = np.quantile(envelope_alpha[good_idx],[0.1,0.9])
    thr_low /=1.7 #2.3 for powerfull alpha 1.5 for low
    thr_high /=1.7#2.5
    
'''




plt.figure()
plt.plot(envelope_alpha[2000:-2000])
#plt.plot(gt_beta_envelope[2000:-2000]*100)
plt.plot(np.real(filtered)[2000:-2000])
plt.plot(np.ones(envelope_alpha[2000:-2000].shape[0])*thr_low,'--',color = 'black')
plt.plot(np.ones(envelope_alpha[2000:-2000].shape[0])*thr_high,'--',color = 'black')
plt.plot(raw_copy[:][0][0,:][2000:-2000])



f,pxx = sn.welch(np.real(filtered[2000:-2000]), nperseg = 2000,fs = srate)
plt.figure()
plt.plot(f,np.log10(pxx))









#mask0 = np.uint32(0x0000FFFF)
#mask1 = np.uint32(0xFFFF0000)

#thr0 = np.bitwise_and(thr, mask0)
#thr1 = np.right_shift(np.bitwise_and(thr, mask1),16)


b_dc_int, a_dc_int = (np.round(0x1000*b_dc)).astype(int),(np.round(0x1000*a_dc)).astype(int)
b50_int, a50_int = (np.round(0x1000*b50)).astype(int),(np.round(0x1000*a50)).astype(int)
b100_int, a100_int = (np.round(0x1000*b100)).astype(int),(np.round(0x1000*a100)).astype(int)
b150_int, a150_int = (np.round(0x1000*b150)).astype(int),(np.round(0x1000*a150)).astype(int)
b200_int, a200_int = (np.round(0x1000*b200)).astype(int),(np.round(0x1000*a200)).astype(int)
b250_int, a250_int = (np.round(0x1000*b250)).astype(int),(np.round(0x1000*a250)).astype(int)

ica_filter_int = np.round(ica_filter_norm*0x1000).astype(int)



print(kf.K)



print(1.0/kf.K)
inv_K = 1.0/kf.K


inv_K_int = (1.0/kf.K.flatten()).astype('int32')

print(inv_K_int)


Phi_int = (kf.Phi.flatten()*0x800).astype(int)
print(Phi_int)








rescale_coef = 20e6*(1<<6) # nedoscaling of envelope on eegboard
thr_low_int = np.array([round(thr_low*rescale_coef)]).astype('int')
thr_high_int=np.array([round(thr_high*rescale_coef)]).astype('int')

print('thresholds:')
print(thr_low_int)
print(thr_high_int)



combined_array = np.concatenate((b_dc_int,a_dc_int,b50_int,a50_int,b100_int,a100_int,b150_int,a150_int,b200_int,a200_int,b250_int,a250_int,ica_filter_int,thr_low_int,thr_high_int,inv_K_int,Phi_int))#,np.array([777])))
print(combined_array)
print(combined_array.shape)


# Write the array to a text file with comma+space delimiters
with open('C:/Users/Fedosov/Documents/projects/mks_return_ICA/FilterTest/Win32/Debug/cfir.txt', 'w') as file:
    file.write(', '.join(str(x) for x in combined_array))
    
    
with open(folder_path+pathdir+'/filter_params.pickle', 'wb') as file:
    
    
    pickle.dump({'freq0': freq_alpha, 'A': A,
            'q': q, 'r': r, 'ica_filter':ica_filter_norm, 'low_thr': thr_low*20e6,'high_thr': thr_high*20e6 }, file = file)





'''

### simulation
envelope_beta_bias = np.roll(envelope_beta,100)

refract_T = 2000
c_counter = 0

high_idx =list()
low_idx = list()
decision = 0
for i in range(envelope_beta_bias.shape[0]):
    if c_counter == 1:
        decision = np.random.randint(2)
        
    if decision ==0:
        if (envelope_beta_bias[i] > thr_high) and(c_counter > refract_T):
            high_idx.append(i)
            c_counter = 0
            
            
    if decision ==1:
        if (envelope_beta_bias[i] < thr_low) and(c_counter > refract_T):
            low_idx.append(i)
            c_counter = 0
        
       
    
    c_counter += 1

high_idx = np.array(high_idx)[2:-2]
low_idx = np.array(low_idx)[2:-2]

high_values = gt_beta_envelope[high_idx]

low_values = gt_beta_envelope[low_idx]


condition1_mean = np.mean(high_values)
condition1_std = np.std(high_values)

condition2_mean = np.mean(low_values)
condition2_std = np.std(low_values)
x = np.arange(2)
labels = ['High', 'Low']

fig, ax = plt.subplots()
ax.bar(x, [condition1_mean, condition2_mean], yerr=[condition1_std, condition2_std], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Values')
ax.set_title('Bar Chart with Standard Deviation')
plt.show()


plt.figure()
plt.scatter(np.zeros(high_values.shape[0])+(np.random.rand(high_values.shape[0])-0.5)/8.0,high_values,c = 'r',s = 50, alpha = 1)

plt.scatter(np.ones(low_values.shape[0])+(np.random.rand(low_values.shape[0])-0.5)/8.0,low_values,c = 'b',s = 50, alpha = 1)
'''

'''
alpha_cc_cum = np.zeros(99)
for i,bias in enumerate(range(1,100)):


    gt_alpha_envelope_trim = gt_alpha_envelope[:-bias]
    gt_beta_envelope_trim = gt_beta_envelope[:-bias]
    
    gt_alpha_envelope_trim -= np.mean(gt_alpha_envelope)
    gt_beta_envelope_trim -= np.mean(gt_beta_envelope)

    
    
    alpha_envelope_trim = alpha_envelope[bias:]
    beta_envelope_trim = beta_envelope[bias:]


    alpha_envelope_trim -= np.mean(alpha_envelope_trim)
    beta_envelope_trim -= np.mean(beta_envelope_trim)
    


    alpha_cc_cum[i] = np.sum(alpha_envelope_trim*gt_alpha_envelope_trim)/(np.linalg.norm(alpha_envelope_trim)*np.linalg.norm(gt_alpha_envelope_trim))
    #beta_cc_cum[i] = np.sum(beta_envelope*gt_beta_envelope)/(np.linalg.norm(beta_envelope)*np.linalg.norm(gt_beta_envelope))


plt.figure()
plt.plot(np.arange(1,100)*1000/srate,alpha_cc_cum)
plt.xlabel('latency, ms')
plt.ylabel('corr')
plt.plot([20,20],[0.55,0.9])

'''






'''

t = np.arange(10000)/1000

signal = np.sin(t*35*2*np.pi)


filtered_signal  = cfir.apply(sig[:,0])#signal)

plt.figure()
plt.plot(np.abs(filtered_signal))

plt.figure()
plt.plot(np.real(filtered_signal))

'''




'''

##### SHIM SIMULATION

nP = 10

b_envelope, a_envelope = sn.butter(1,1.0,btype = 'low',fs = 1000)


env_smoothed = sn.lfilter(b_envelope, a_envelope,envelope_beta)
plt.figure()
#plt.plot(envelope_beta)
plt.plot(env_smoothed)


env_max,env_min = np.quantile(env_smoothed[good_idx], [0.95,0.05])
env_nT = env_smoothed.shape[0]


env_shim = np.zeros(env_nT)
counter = 0
duty_counter = 0

for i in range(env_nT):
    if counter % nP == 0:
        
        sample = env_smoothed[i]
        if(env_smoothed[i]>env_max):
            sample = env_max
        if(env_smoothed[i]<env_min):
            sample = env_min
        
        
        

        duration = ((sample-env_min)*nP)/(env_max-env_min)
        duty_counter = 0
    if duty_counter < duration:
        env_shim[i] = 1
    else:
        env_shim[i] = 0
    
            
    
    
    duty_counter += 1
    counter += 1
    


b_shim, a_shim = sn.butter(1,1.0,btype = 'low',fs = 1000)

env_recon = sn.lfilter(b_shim,a_shim,env_shim)


plt.figure()

plt.plot((env_shim)[2000:])
plt.plot((env_smoothed/env_max)[2000:])
plt.plot(env_recon[2000:])


plt.legend(['original','shim','recon'])
    



'''






