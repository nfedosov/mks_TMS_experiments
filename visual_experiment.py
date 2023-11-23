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
import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5 import QtCore
from pyqtgraph import PlotWidget
import pyqtgraph as pg



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



# класс отрисовки



class DataProcessor():
    
    def __init__(self):
        
        
        exp_settings = {
            #максимальная буферизация принятых данных через lsl
            'max_buflen': 5,  # in seconds!!!!
            #максимальное число принятых семплов в чанке, после которых появляется возможность
            #считать данные
            'max_chunklen': 1,  # in number of samples!!!!
            'lsl_stream_name': 'NVX52_Data'
            }
        self.inlet = LSLInlet(exp_settings)
        
        
        #!!!!
        self.srate= 1000
        self.n_channels = 4
        
        
        self.b_dc, self.a_dc = sn.butter(1,5.0,btype = 'high',fs = self.srate)
        self.z_dc = np.zeros(1)
        
        #self.b_low, self.a_low = sn.butter(1,50.0,btype = 'low',fs = self.srate)
        #self.z_low = np.zeros(1)
        
        self.b50,self.a50=sn.butter(1,[48.0,52.0], btype = 'bandstop',fs = self.srate)
        self.z_50 = np.zeros(2)
        
        self.b100,self.a100=sn.butter(1,[97,103], btype = 'bandstop',fs = self.srate)
        self.z_100 = np.zeros(2)
        
        self.b150,self.a150 = sn.butter(1,[146.0,154.0], btype = 'bandstop',fs = self.srate)
        self.z_150 = np.zeros(2)
        
        self.b200,self.a200 = sn.butter(1,[195.0,205.0], btype = 'bandstop',fs = self.srate)
        self.z_200 = np.zeros(2)
        
        self.b250,self.a250 = sn.butter(1,[244.0,256.0], btype = 'bandstop',fs = self.srate)
        self.z_250 = np.zeros(2)
        
        
        self.b_smooth, self.a_smooth = sn.butter(1,0.31,btype = 'low',fs = self.srate)
        self.z_smooth = np.zeros(1)
         

        self.load_params()
        
        self.kf = float_int_kalman(self.freq0,self.A, self.srate,self.q,self.r)
        
        #max 10 min
        self.data= np.zeros((1000*1000,self.n_channels))
        
        self.n_samples_received = 0

        
    
    #def process_chunk(self,chunk):
    #    pass
        
    def load_params(self):
                
        folder_path = 'C:/Users/Fedosov/Documents/projects/AudioNFB/results/'
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
        
        self.freq0 = container['freq0']
        self.A = container['A']
        self.q = container['q']
        self.r = container['r']
        self.ica_filter = container['ica_filter']
        self.low_thr = container['low_thr']
        self.high_thr = container['high_thr']
        
        self.envelope = np.ones(1)*self.low_thr
                
        
    def iteration(self):
        chunk, t_stamp = self.inlet.get_next_chunk()
        # print(f"{chunk=}")
        
        if chunk is not None:
            n_samples_in_chunk = len(chunk)
            self.data[self.n_samples_received:self.n_samples_received + n_samples_in_chunk, :] = chunk
            
            
            chunk = self.ica_filter@(chunk.T)
            
            chunk, self.z_dc = sn.lfilter(self.b_dc,self.a_dc,chunk, zi = self.z_dc)
            
            chunk, self.z_50 = sn.lfilter(self.b50,self.a50,chunk, zi = self.z_50)
            chunk, self.z_100 = sn.lfilter(self.b100,self.a100,chunk, zi = self.z_100)
            chunk, self.z_150 = sn.lfilter(self.b150,self.a150,chunk, zi = self.z_150)
            chunk, self.z_200 = sn.lfilter(self.b200,self.a200,chunk, zi = self.z_200)
            chunk, self.z_250 = sn.lfilter(self.b250,self.a250,chunk, zi = self.z_250)
            
            filtered, _, self.envelope = self.kf.apply(chunk)
            
            tosmooth = 1
            if tosmooth:
                self.envelope, self.z_smooth = sn.lfilter(self.b_smooth,self.a_smooth,self.envelope,zi = self.z_smooth)
             
            print(self.high_thr)
            
            self.n_samples_received += n_samples_in_chunk
          

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NFB")
  

        # Set up plot widget
        self.plot_widget = PlotWidget()
        self.dataprocessor = DataProcessor()
        self.plot_widget.setYRange(self.dataprocessor.low_thr,self.dataprocessor.high_thr)  # Adjust the y-range as needed
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        
        # Set fixed window size
        self.setFixedSize(600, 600)  # Adjust size as needed

        # Set up layout and central widget
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


        self.additional_window = AdditionalWindow()
        self.additional_window.showMaximized() 
        # Start a timer to update the plot periodically
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10)  # Update every 10 milliseconds (adjust as needed)
        

    def update_plot(self):
      #np.ran\
        # Update plot
        self.dataprocessor.iteration()
        self.plot_widget.clear()
        #[self.dataprocessor.envelope[-1]]
        if self.dataprocessor.envelope[-1] < self.dataprocessor.low_thr:
            self.dataprocessor.envelope[-1] = self.dataprocessor.low_thr
        if self.dataprocessor.envelope[-1] > self.dataprocessor.high_thr:
            self.dataprocessor.envelope[-1] = self.dataprocessor.high_thr
        self.plot_widget.addItem(pg.BarGraphItem(x=[0], height=[self.dataprocessor.envelope[-1]], width=0.7, brush = [0xD3,0x4d,0xD2]),pen=None)  # Adjust width as needed
        #print(self.dataprocessor.envelope[-1])
    




class AdditionalWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Additional Window")
        self.setGeometry(100, 100, 400, 400)

        # Create a label to hold the image
        label = QLabel(self)
        pixmap = QPixmap("paths-transformed.jpeg")
        label.setPixmap(pixmap)
        label.resize(pixmap.width(), pixmap.height())

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    
    window.show()
    app.exec()


