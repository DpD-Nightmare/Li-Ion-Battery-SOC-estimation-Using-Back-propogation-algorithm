'''
This program is for loading the data files ie(Excel,csv,xlsx,...)
Pandas function is used for loading data files only then all the 
math operaton is done with .numpy
.matplotlib operater is ued for ploting the datas
SMA filter is used for filter the data. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve

#Importing Data files
fname = "Traing_battery_DST.xlsx"
data= pd.read_excel("Traing_battery_DST.xlsx")

#Define of SMA
def movingavrage(value, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(value, weights,'valid')
    return sma

#data= data.sheet_by_index(0)
a=data['Current(A)'] #Importing Current data from data sheet
curr = movingavrage (a,10) #Filter the data with  window of 10
curr = list(curr) #Declaring the List

#print(len(curr))
plt.plot(a)
plt.plot(curr)
#amp= np.array([curr])
#print('old matrix',amp.shape)
#print(curr)
#irate= amp.reshape(26665,1)
#print(irate[15000])
#print('new matrix',irate.shape)
#print(irate)

v=data['Voltage(V)'] #Imporint Voltage data from data sheet
vol = movingavrage (v,10) #Filtering the data
vol = list(vol)

#volte= np.array([vol])
#vrate= volte.reshape(26665,1)
#print(vrate[15000])
#print('Live volt',vrate.shape)
#print(vrate)

t=data['Temp(C)'] #Imporitng Temprature data from data sheet
te= movingavrage (t,10) #Filtering the data
te= list(te)

#tem= np.array([te])
#trate= tem.reshape(26665,1)
#print(trate[15000])
#rint(trate.shape)

ah=data['cap(Ah)'] #Importing the Charge capacity form data sheet
ch= movingavrage (ah,10) #Filtering the data
ch= list(ch)
cha=np.array([ch]) #Converting the list value in to Matrix form
charge= cha.reshape(26665,1) #Transposed the Matrix for further calculation

#print(charge.shape)
#print(charge[15000])
#print('desierd charge',charge.shape)
#print(charge)

data_in =np.array([curr,vol,te]) #Creating Input data to the Neural Network I,V,T
data_in = data_in.T #Transpose the NN_input datas
print(data_in.shape)
print(data_in[15000])

#print(data_in)
#cl = list(data_in.columns)
#print(cl.head())
#print('Input data matrix',data_in.shape)

data_ref = np.array(charge) #Creating the Refrance output datas for NN SOC'''
print('refrance data matrix',data_ref.shape)

#plt.plot(a)
#plt.plot(v)
#plt.plot(t)
#plt.plot(ah)

#plt.plot(data_in)
#plt.plot(data_ref)
plt.show()