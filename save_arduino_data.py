# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:03:48 2023

@author: Fedosov
"""

import serial

# Replace 'COMX' with the  actual name of your Arduino's serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
serial_port = serial.Serial('COM16', 115200)  # Use the same baud rate as in your Arduino sketch

# Define the file where you want to save the data
output_file = open('arduino_data.txt', 'w')

try:
    while True:
        # Read data from Arduino
        try:
            data = serial_port.readline().decode().strip()
        except:
            pass

        # Print and save the data to a file
        print(data)
        output_file.write(data + '\n')

except KeyboardInterrupt:
    print("Interrupted")
    output_file.close()
    serial_port.close()
    
    
    
    
    
    