'''
#Program for testing all kind of
timeseries Neural network avlable in 
tensorflow and keras lib

#cheak what are the changes we can 
do in the avilabel tensor neuranetworks

#Tharaly See relation between 
learning rate, number of hidden layer, 
and initaly set weights.
'''
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from numpy import convolve
start_time=time.time()
'''
Defination section
'''
#Moving Avrage
def movingavrage (value, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(value, weights, 'valid')
    return sma

#Normalization of Data
def norm(x):
    return(x -x_stats['min']) / (x_stats['max'] - x_stats['min'])

'''
Imporing statstical data in program
'''

fname = pd.read_excel('LFP_007_mi8.xlsx', 'Channel_1-006')
#print(fname.head())

'''
Take the important data for NN from data sheet
'''
cols = ['I', 'V', 'T', 'dVdt', 'didt', 'Wh']
new_data = pd.DataFrame(index=range(len(fname)),columns= cols)
new_data['I'] = fname['Current(A)']
new_data['V'] = fname['Voltage(V)']
new_data['dVdt'] = fname['dV/dt(V/s)']
new_data['T'] = fname['Temperature (C)_1']
new_data['didt'] = fname['Changed(di_dt)']
new_data['Wh'] = fname['Capacity(Wh)']
#print(new_data.head())

x_stats = new_data.describe()
x_stats.pop('Wh')
x_stats = x_stats.transpose()
#print(x_stats)

'''Training batches'''
x_t = new_data.copy()
y_t = x_t.pop('Wh')

x0_t = np.array(norm(x_t[0:6855]))
y0_t = np.array(y_t[0:6855])

xtest_t = np.array(norm(x_t[5788:19000]))
yte_t = np.array(y_t[5788:19000])
print(x0_t)
print("")
print(y0_t)

'''Neural network'''

model = Sequential()
model.add(Dense(18,activation = 'sigmoid', input_shape = [x0_t.shape[1]]))
#model.add(Dense(120, activation= 'sigmoid', input_shape = [x0_t.shape[1]]))
model.add(Dense(18, activation='sigmoid'))
model.add(Dense(18, activation= 'sigmoid'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
#print(model.summary())
history = model.fit(x0_t,y0_t, epochs= 10000, batch_size= 100)
#print('')
print(history)
print("")
#print('loss')
stop_time = time.time()-start_time
print("Training Time = ",stop_time)
y_prid = model.predict(x0_t)
plt.plot(y_prid, label='predect')
plt.plot(y0_t, label = 'actual')

#plt.plot(loss, label= 'loss function')
plt.legend()
plt.show()
