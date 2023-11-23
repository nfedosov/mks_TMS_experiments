# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 08:33:35 2023

@author: Fedosov
"""

import numpy as np



seq = np.array([],dtype = 'int')

seq = np.concatenate([seq, np.zeros([3],dtype ='int')])
for i in range(10):
    new_seq =np.arange(0,18, dtype = 'int')
    np.random.shuffle(new_seq)
    seq = np.concatenate([seq, new_seq])
    

    

print(np.array2string(seq, separator=","))

print(len(seq))