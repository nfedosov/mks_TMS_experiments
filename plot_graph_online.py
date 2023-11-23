# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:39:23 2023

@author: Fedosov
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pylsl

def receive_data():
    # Connect to LSL stream
    streams = pylsl.resolve_stream('type', 'your_stream_type', timeout=5)
    inlet = pylsl.StreamInlet(streams[0])
    
    # Set up the plot
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Multichannel Plot")
    win.resize(800, 600)
    plots = []
    
    # Create a plot for each channel
    for i in range(inlet.channel_count()):
        plot = win.addPlot(title=f"Channel {i+1}")
        plot.showGrid(x=True, y=True)
        plots.append(plot)
    
    # Start receiving and plotting data
    while True:
        # Read a chunk of data from the LSL stream
        chunk, timestamps = inlet.pull_chunk()
        
        # Update each plot with the new data
        for i, data in enumerate(chunk):
            x = timestamps
            y = data
            plots[i].plot(x, y)
        
        # Process Qt events to keep the GUI responsive
        QtGui.QApplication.processEvents()
    
    # Start the Qt event loop
    QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    receive_data()
