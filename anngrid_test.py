import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''Math library'''
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
start_time = time.time()
#data normalization
'''data'''
fname = pd.read_excel('LFP_007_mi8.xlsx', 'Channel_1-006')
#print(fname)

#Take the important data for NN from data sheet

cols = ['I', 'V', 'T', 'dVdt', 'didt', 'Wh']
new_data = pd.DataFrame(index=range(len(fname)),columns= cols)
new_data['I'] = fname['Current(A)']
new_data['V'] = fname['Voltage(V)']
new_data['dVdt'] = fname['dV/dt(V/s)']
new_data['T'] = fname['Temperature (C)_1']
new_data['didt'] = fname['Changed(di_dt)']
new_data['Wh'] = fname['Capacity(Wh)']
#print(new_data.head())

#print(new_x)
new_x = new_data
x0 = new_x.iloc[:, 0:4].values
y0 = new_x.iloc[:, 5].values
'''
#plt.plot(y0, label = 'Refrance')
#plt.legend()
#plt.show()

#DST test for -10C
#x = x0[1545:6855]
#y = y0[1545:6855]


#US06 test for -10C
#x = x0[7920:13050]
#y = y0[7920:13050]

#FUDs test for -10C
x = x0[14100:]
y = y0[14100:]
'''
#Test at -10C full data
x = x0[0:]
y= y0[0:]
#spliting the training and testing datas
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#scaling of data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print('training Inputs', x_train)
#print('Tetsting datas', x_test)

#ANN model
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_model(optimizer):
    model = Sequential()
    model.add(Dense(output_dim=18, init= 'uniform', activation = 'sigmoid', input_dim =4))
    model.add(Dense(output_dim=18, init= 'uniform', activation = 'sigmoid'))
    model.add(Dense(output_dim=18, init= 'uniform', activation = 'sigmoid'))
    model.add(Dense(output_dim=1))
    model.compile(optimizer = optimizer, loss= 'mse', metrics = ['mae'])
    return model

model = KerasRegressor(build_fn= build_model)
parameters = {'batch_size':[100,150],
              'epochs': [10000,12500,11500],
              'optimizer':['adam']}
grid_search= GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring= 'neg_mean_squared_error',n_jobs=-1, cv=5)
grid_search= grid_search.fit(x_train, y_train)
best_perameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('best_perameters for FUDS',best_perameters)
print('best_MSE', best_accuracy)
stop_time = time.time()-start_time
print("Optimizing time =",stop_time)

