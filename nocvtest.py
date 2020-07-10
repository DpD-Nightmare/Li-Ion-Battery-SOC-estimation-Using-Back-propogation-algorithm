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
'''Math library'''
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from numpy import convolve
start_time = time.time()

'''data at 25C for training'''
fname = pd.read_excel('A1-007-25.xlsx', 'Channel_1-006')
#print(fname)

#Take the important data for NN from data sheet

cols = ['I', 'V', 'T', 'dVdt', 'didt', 'charg', 'discharg', 'dst', 'us06','fuds', 'Wh']

new_data = pd.DataFrame(index=range(len(fname)),columns= cols)

new_data['I'] = fname['Current(A)']
new_data['V'] = fname['Voltage(V)']
new_data['dVdt'] = fname['dV/dt(V/s)']
new_data['T'] = fname['Temperature (C)_1']
new_data['didt'] = fname['Changed(di_dt)']
new_data['charg'] = fname['charging']
new_data['discharg'] = fname['discharging']
new_data['dst'] = fname['DST']
new_data['us06'] = fname['US06']
new_data['fuds'] = fname['FUDS']
new_data['Wh'] = fname['Capacity(Wh)']
#print(new_data)
print(new_data.head())

new_x = new_data
x0 = new_x.iloc[:, 0:10].values
y0 = new_x.iloc[:, 10].values
#print('in',x0)

#Training network with 25C full data
x = x0[0:]
y= y0[0:]
#print('in',x0)

#plt.plot(y, label = 'actual')


#print('new',x)
#spliting the training and testing datas
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#z = y_train.transpose()
#plt.plot(z, label = 'train')
#plt.legend()
#plt.show()

#scaling of data in range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print('training Inputs', x_train)
#print('Tetsting datas', x_test)

#ANN model
import keras
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_validate
from keras.models import Sequential
from keras.layers import Dense

#Neural network

model = Sequential()
model.add(Dense(18,activation = 'sigmoid', input_dim = 10))
#model.add(Dense(120, activation= 'sigmoid', input_shape = [x0_t.shape[1]]))
model.add(Dense(18, activation='sigmoid'))
model.add(Dense(18, activation= 'sigmoid'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
#print(model.summary())
history = model.fit(x_train,y_train, epochs= 10000, batch_size= 150)
#print('')
print(history)
print("")
#print('loss')
stop_time = time.time()-start_time
print("Training Time = ",stop_time)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
y_prid = model.predict(x)
plt.plot(y_prid, label='predect')
plt.plot(y, label = 'actual')

#plt.plot(loss, label= 'loss function')
plt.legend()
plt.show()

