# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 00:06:31 2016

@author: ilia
"""

import os
os.chdir('/home/ilia/Source/CudaCourse/Problem Sets/Problem Set 6')

import numpy as np
with open('num_neighbors.txt','r') as f:
    X=np.loadtxt(f,delimiter=',')

from matplotlib import pyplot as plt
plt.figure()
plt.imshow(X)
plt.colorbar()


