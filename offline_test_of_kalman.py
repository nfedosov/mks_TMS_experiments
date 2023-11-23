# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:41:24 2023

@author: Fedosov
"""



import numpy as np
import scipy.signal as sn
import matplotlib.pyplot as plt
import scipy.stats as st

plt.close('all')

class BaseKalman:
    def __init__(self, H, Phi, Q, R):
        
        self.x = np.zeros((Phi.shape[0],1))
        self.P = np.eye(Phi.shape[0])
        
        self.Q = Q
        self.R = R
        self.H = H
        self.Phi = Phi
        
        
        
    def step(self, y):
        self.x_ = self.Phi@self.x
        
        self.P_ = self.Phi@self.P@self.Phi.T+self.Q
        
        self.res = y-self.H@self.x_
        
        self.S = self.H@self.P_@self.H.T+self.R
        
        self.K = self.P_@self.H.T/self.S
        
        self.x = self.x_+self.K@self.res
        
        self.P = (np.eye(self.P.shape[0])-self.K@self.H)@self.P_
        #print(self.P[0,0])
        
        
    
        
        
        
class WhiteKalman(BaseKalman):
    def __init__(self, freq0, A, srate, q, r):
        
        self.freq0 = freq0
        self.A = A
        H = np.array([[1.0,0.0]])
        Phi = A*np.array([[np.cos(2.0*np.pi*freq0/srate),-np.sin(2.0*np.pi*freq0/srate)],[np.sin(2.0*np.pi*freq0/srate),np.cos(2.0*np.pi*freq0/srate)]])
        
        self.Q = np.eye(2)*q
        self.R = r
        
        
        
        super().__init__(H, Phi,self.Q, self.R)
        
        
   
    def apply(self, y):
        
        Len = y.shape[0]
        filtered_real =np.zeros(Len)
        filtered_imag = np.zeros(Len)
        envelope =np.zeros(Len)
        phase = np.zeros(Len)
        complex_signal = np.zeros(Len, dtype = 'complex')
        
        for i in range(Len):
            self.step(y[i])
            #filtered_real[i] = self.x[0]
            #filtered_imag[i] = self.x[1]
            #envelope[i] = np.sqrt(self.x[0]**2+self.x[1]**2)
            #phase[i] = np.angle(self.x[0]+self.x[1]*1j)
            complex_signal[i] = self.x[0]+self.x[1]*1j
            
        return complex_signal#,filtered_real, filtered_imag, envelope, phase
       
    
    
    
    
    
    
 
        
class SignalGenerator(BaseKalman):
    def __init__(self, freq0, A, srate, q, r):
        
        self.freq0 = freq0
        self.A = A
        self.H = np.array([[1.0,0.0]])
        self.Phi = A*np.array([[np.cos(2.0*np.pi*freq0/srate),-np.sin(2.0*np.pi*freq0/srate)],[np.sin(2.0*np.pi*freq0/srate),np.cos(2.0*np.pi*freq0/srate)]])
        
        self.Q = np.eye(2)*q
        self.R = r
        
        
        
        
  
   
    def generate(self, nT, smooth = False):
        
        self.x = np.zeros((2,1))
        
        
        generated_signal = np.zeros(nT)
        
        real_part =np.zeros(nT)
        imag_part = np.zeros(nT)
        phase_gt = np.zeros(nT)
        
        for i in range(nT):
        
            real_part[i] = self.x[0,0]
            imag_part[i] = self.x[1,0]
            
            #phase_gt[i] = np.angle(self.x[0,0]+1j*self.x[1,0])
            
            self.x = self.Phi@self.x+np.random.randn(2,1)*self.Q
          
        if smooth == False:
            phase_gt = np.angle(real_part + 1j*imag_part)
            
        else:
            b, a = sn.butter(2,20.0,btype = 'low',fs = srate)
            real_part = sn.lfilter(b,a,real_part)
            
            hilb = sn.hilbert(real_part)
            phase_gt = np.angle(hilb)
            
        generated_signal = real_part + np.random.randn(nT)*self.R
    
        
        
        return generated_signal, phase_gt#,filtered_real, filtered_imag, envelope, phase
       
    
    
    
    
    
freq0 = 10.0
srate = 1000
A = 0.995
q = 0.01#100#0.01
r = 100#0.01#100

q_gen = 100.0
r_gen = 500.0#0.01

white_kf = WhiteKalman(freq0, A, srate, q, r)

signal_generator = SignalGenerator(freq0, A, srate, q_gen, r_gen)



nT = 100000

generated_signal, phase_gt = signal_generator.generate(nT, smooth = True)

plt.figure()
plt.plot(generated_signal)





period = 1000



picked_phases_gt = phase_gt[::period]




cmplx_filtered = white_kf.apply(generated_signal)

plt.plot(np.real(cmplx_filtered))
#cmplx_filtered = np.roll(cmplx_filtered,10)


phase = np.angle(cmplx_filtered)


picked_phases = phase[::period]



plt.figure()
plt.plot(picked_phases)
plt.plot(picked_phases_gt)







def phase_difference(angle1, angle2):
    # Ensure angles are in the range [0, 2π]
    #angle1 = angle1 % (2 * math.pi)
    #angle2 = angle2 % (2 * math.pi)

    # Calculate the absolute angular difference
    abs_diff = abs(angle1 - angle2)

    # Consider the "wrap-around" case when angles are close to 0 and 2π
    abs_diff[abs_diff>180] = 360-abs_diff[abs_diff>180]
    

    return abs_diff


phase_dif = phase_difference(picked_phases_gt*180/np.pi, picked_phases*180/np.pi)#st.circstd(target_phases, real_phases)



std_dif = st.circstd(phase_dif*np.pi/180)

plt.figure()
plt.hist(phase_dif,bins = 20, range = [0, 180])
plt.xlabel('phase dif')
plt.title('strong kalman, r_gen/q_gen = 5.0')


sign_phase_dif = picked_phases-picked_phases_gt

plt.figure()
plt.hist(sign_phase_dif*180/np.pi,bins = 20, range = [-360, 360])























