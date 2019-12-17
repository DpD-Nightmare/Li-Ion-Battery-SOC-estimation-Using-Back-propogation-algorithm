#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:50:16 2019

@author: dhrupad
"""

import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt

def movingavrage (value, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(value, weights,'valid')
    return sma

x = [1,2,3,4,5,6,7,8,9,10]
y = [3,5,1,8,2,1,6,1,9,2]

xMA = movingavrage (x,3)
yMA = movingavrage (y,3)
yMA = list(yMA)
yMA.insert(0,y[0])
yMA.insert(1,y[1])
print(len(yMA))
print(yMA)
print (xMA)
plt.plot(x,y)
plt.plot(x,yMA)
#plt.show()