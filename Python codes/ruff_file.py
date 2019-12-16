#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:50:16 2019

@author: dhrupad
"""

import numpy as np
import pandas as pd

product = {'month' : [1,2,3,4,5,6,7,8,9,10,11,12], 'demand':[200,120,150,600,450,650,400,290,300,100,700,850]}

df= pd.DataFrame(product)

df.reshape(2,12)
print(df.shape)